from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .mmd_abc import median_heuristic_bandwidth
from .upv_phase3d import (
    _apply_route_means,
    _balanced_scale,
    _evaluate_generator,
    _feature_sequences,
    _fit_route_means,
    _git_revision,
    _joint_mmd,
    _load_development_sessions,
    _read_json,
    _read_yaml,
    _real_pairwise_reference,
    _sample_blocks,
    _sha256,
    _temporal_error,
    _write_csv,
    _write_json,
    validate_phase3d_config,
)


def validate_phase3f_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3F schema_version must be 1")
    if config.get("stage") != "phase_3f_development_exchangeability_diagnostic":
        raise ValueError("unexpected Phase 3F stage")
    if any(
        bool(config.get(name))
        for name in ("execution_authorized", "final_evaluation_authorized", "abc_authorized")
    ):
        raise ValueError("Phase 3F cannot authorize execution, final access, or ABC")
    frozen = config.get("frozen_inputs") or {}
    for name in ("archive_sha256", "phase3d_config_sha256", "phase3e_result_sha256"):
        value = str(frozen.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid Phase 3F frozen checksum: {name}")
    if frozen.get("required_phase3e_decision") != "temporal_process_revision_still_unsupported":
        raise ValueError("Phase 3F requires the frozen Phase 3E rejection")
    development = config.get("development") or {}
    if development.get("cross_validation") != "leave_one_complete_session_out":
        raise ValueError("Phase 3F requires leave-one-session-out development")
    if int(development.get("folds_required", 0)) != 5:
        raise ValueError("Phase 3F requires five folds")
    if development.get("final_test6_access") is not False:
        raise ValueError("Phase 3F must keep Test 6 inaccessible")
    oracle = (config.get("diagnostics") or {}).get("holdout_fitted_oracle") or {}
    if oracle.get("selectable_process") is not False:
        raise ValueError("the holdout-fitted oracle cannot be selected")
    if int(oracle.get("block_length_rows", 0)) != 60:
        raise ValueError("the frozen diagnostic oracle uses 60-row blocks")
    if int(oracle.get("generation_repetitions", 0)) < 20:
        raise ValueError("the diagnostic oracle requires at least twenty repetitions")
    rules = config.get("decision_rules") or {}
    if int(rules.get("required_exchangeable_folds", 0)) != 4:
        raise ValueError("the exchangeability gate must require four folds")
    if int(rules.get("required_oracle_supported_folds", 0)) != 4:
        raise ValueError("the oracle gate must require four folds")
    claims = config.get("claim_limits") or {}
    if any(value != "prohibited" for value in claims.values()):
        raise ValueError("all Phase 3F claim limits must remain prohibited")
    reservation = config.get("reservation") or {}
    if reservation.get("request_now") is not False:
        raise ValueError("Phase 3F cannot request a reservation")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must remain at least 30 minutes")


def phase3f_decision(
    *, exchangeable_folds: int, oracle_supported_folds: int, config: dict[str, Any]
) -> dict[str, Any]:
    rules = config["decision_rules"]
    if exchangeable_folds < int(rules["required_exchangeable_folds"]):
        outcome = rules["outcomes"]["cross_session_shift"]
    elif oracle_supported_folds < int(rules["required_oracle_supported_folds"]):
        outcome = rules["outcomes"]["unattainable_metric"]
    else:
        outcome = rules["outcomes"]["model_failure"]
    return {
        "decision_code": outcome["code"],
        "next_action": outcome["next_action"],
        "exchangeable_folds": exchangeable_folds,
        "oracle_supported_folds": oracle_supported_folds,
        "process_selection_authorized": False,
        "powder_reservation_authorized": False,
        "final_evaluation_authorized": False,
        "abc_authorized": False,
    }


def analyze_phase3f_exchangeability(
    *,
    archive_path: str | Path,
    phase3d_config_path: str | Path,
    phase3e_result_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    archive = Path(archive_path).resolve()
    phase3d_file = Path(phase3d_config_path).resolve()
    phase3e_file = Path(phase3e_result_path).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3F output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3f_config(config)
    phase3d = _read_yaml(phase3d_file)
    validate_phase3d_config(phase3d)
    frozen = config["frozen_inputs"]
    observed_checksums = {
        "archive_sha256": _sha256(archive),
        "phase3d_config_sha256": _sha256(phase3d_file),
        "phase3e_result_sha256": _sha256(phase3e_file),
    }
    for name, value in observed_checksums.items():
        if value != frozen[name]:
            raise ValueError(f"Phase 3F frozen input checksum mismatch: {name}")
    phase3e = _read_json(phase3e_file)
    if phase3e.get("decision_code") != frozen["required_phase3e_decision"]:
        raise ValueError("unexpected Phase 3E result")
    if (phase3e.get("final_evaluation") or {}).get("payload_opened") is not False:
        raise ValueError("the final payload must remain unopened")
    sessions, quality = _load_development_sessions(archive, phase3d)
    if phase3d["final_evaluation"]["source_path"] in set(quality["source_path"]):
        raise AssertionError("the final evaluation payload entered Phase 3F")

    diagnostics = config["diagnostics"]
    oracle_config = diagnostics["holdout_fitted_oracle"]
    lags = [
        int(value) for value in diagnostics["inherited_metrics"]["autocorrelation_lags_seconds"]
    ]
    maximum_rows = int(diagnostics["inherited_metrics"]["maximum_rows_per_trace"])
    block_length = int(oracle_config["block_length_rows"])
    repetitions = int(oracle_config["generation_repetitions"])
    seed = int(oracle_config["seed"])
    direct_rows = []
    fold_rows = []
    session_ids = sorted(sessions)
    for fold_index, holdout_id in enumerate(session_ids, start=1):
        fold_id = f"fold_{fold_index}_{holdout_id}"
        training = {key: value for key, value in sessions.items() if key != holdout_id}
        route_means = _fit_route_means(training, phase3d)
        holdout, support = _apply_route_means(sessions[holdout_id], route_means, phase3d)
        if support["unsupported_fraction"] > float(
            phase3d["decision_rules"]["maximum_unsupported_position_fraction_per_fold"]
        ):
            raise ValueError(f"holdout spatial support failed: {fold_id}")
        center, scale = _balanced_scale(training)
        training_targets = np.vstack(
            [frame[["relative_rsrp_db", "sinr_db"]].to_numpy(float) for frame in training.values()]
        )
        bandwidth = median_heuristic_bandwidth((training_targets - center) / scale)
        joint_reference, temporal_reference = _real_pairwise_reference(
            training, center, scale, bandwidth, maximum_rows, lags
        )
        observed = _feature_sequences(holdout, ["relative_rsrp_db", "sinr_db"])
        pair_joint = []
        pair_temporal = []
        for training_id, training_frame in sorted(training.items()):
            training_values = _feature_sequences(training_frame, ["relative_rsrp_db", "sinr_db"])
            joint = _joint_mmd(observed, training_values, center, scale, bandwidth, maximum_rows)
            temporal = _temporal_error(observed, training_values, scale, lags)
            pair_joint.append(joint)
            pair_temporal.append(temporal)
            direct_rows.append(
                {
                    "fold_id": fold_id,
                    "holdout_session_id": holdout_id,
                    "training_session_id": training_id,
                    "joint_mmd_squared": joint,
                    "temporal_error": temporal,
                    "joint_reference_p90": joint_reference,
                    "temporal_reference_p90": temporal_reference,
                }
            )

        residual_holdout = _feature_sequences(holdout, ["rsrp_residual_db", "sinr_residual_db"])
        route_sequences = _feature_sequences(holdout, ["route_relative_rsrp_db", "route_sinr_db"])
        oracle = _evaluate_generator(
            observed=observed,
            route_sequences=route_sequences,
            generator=lambda lengths, rng, source=residual_holdout: _sample_blocks(
                source, lengths, block_length, rng
            ),
            repetitions=repetitions,
            seed=seed + fold_index * 100000,
            center=center,
            scale=scale,
            bandwidth=bandwidth,
            maximum_rows=maximum_rows,
            lags=lags,
        )
        median_joint = float(np.median(pair_joint))
        median_temporal = float(np.median(pair_temporal))
        joint_exchangeable = median_joint <= joint_reference
        temporal_exchangeable = median_temporal <= temporal_reference
        oracle_joint = oracle["joint_mmd_squared_mean"] <= joint_reference
        oracle_temporal = oracle["temporal_error_mean"] <= temporal_reference
        fold_rows.append(
            {
                "fold_id": fold_id,
                "holdout_session_id": holdout_id,
                "direct_median_joint_mmd_squared": median_joint,
                "direct_median_temporal_error": median_temporal,
                "joint_reference_p90": joint_reference,
                "temporal_reference_p90": temporal_reference,
                "joint_exchangeable": joint_exchangeable,
                "temporal_exchangeable": temporal_exchangeable,
                "joint_and_temporal_exchangeable": joint_exchangeable and temporal_exchangeable,
                "oracle_joint_mmd_squared_mean": oracle["joint_mmd_squared_mean"],
                "oracle_temporal_error_mean": oracle["temporal_error_mean"],
                "oracle_joint_supported": oracle_joint,
                "oracle_temporal_supported": oracle_temporal,
                "oracle_joint_and_temporal_supported": oracle_joint and oracle_temporal,
            }
        )
    folds = pd.DataFrame(fold_rows)
    exchangeable_folds = int(folds["joint_and_temporal_exchangeable"].sum())
    oracle_supported_folds = int(folds["oracle_joint_and_temporal_supported"].sum())
    decision = phase3f_decision(
        exchangeable_folds=exchangeable_folds,
        oracle_supported_folds=oracle_supported_folds,
        config=config,
    )
    result = {
        "schema_version": 1,
        "stage": "phase_3f_development_exchangeability_result",
        "protocol_revision": config["protocol_revision"],
        "input_sha256": observed_checksums,
        "analysis_repository": _git_revision(),
        "development_sessions": session_ids,
        "development_folds": len(session_ids),
        "diagnostic_oracle_selectable": False,
        "final_evaluation": {
            "source_path": phase3d["final_evaluation"]["source_path"],
            "payload_opened": False,
            "authorized": False,
        },
        **decision,
        "claim_limits": config["claim_limits"],
        "reservation": config["reservation"],
    }
    output.mkdir(parents=True)
    _write_json(output / "phase3f_decision.json", result)
    _write_csv(output / "real_holdout_training_pairs.csv", pd.DataFrame(direct_rows))
    _write_csv(output / "fold_exchangeability.csv", folds)
    _write_csv(output / "development_input_quality.csv", quality)
    _write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": 1,
            "stage": "phase_3f_analysis_manifest",
            "folds": len(session_ids),
            "direct_real_pairs": len(direct_rows),
            "oracle_generation_repetitions_per_fold": repetitions,
            "final_evaluation_payload_opened": False,
            "reservation_requested": False,
        },
    )
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": decision["decision_code"],
        "exchangeable_folds": str(exchangeable_folds),
        "oracle_supported_folds": str(oracle_supported_folds),
        "final_test6_accessed": "false",
        "reservation_requested": "false",
    }
