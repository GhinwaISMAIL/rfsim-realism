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
    _fit_hmm,
    _fit_route_means,
    _git_revision,
    _load_development_sessions,
    _model_record,
    _read_json,
    _read_yaml,
    _real_pairwise_reference,
    _sample_blocks,
    _sequence_posteriors,
    _sha256,
    _write_csv,
    _write_json,
    validate_phase3d_config,
)


def validate_phase3e_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Phase 3E schema_version must be 1")
    if config.get("stage") != "phase_3e_offline_temporal_process_revision":
        raise ValueError("unexpected Phase 3E stage")
    if any(
        bool(config.get(name))
        for name in ("execution_authorized", "abc_authorized", "final_evaluation_authorized")
    ):
        raise ValueError("Phase 3E cannot authorize execution, ABC, or final access")

    frozen = config.get("frozen_inputs") or {}
    for name in (
        "phase3d_config_sha256",
        "phase3d_decision_sha256",
        "corrected_noise_result_sha256",
        "archive_sha256",
    ):
        value = str(frozen.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"invalid Phase 3E frozen checksum: {name}")
    if frozen.get("required_phase3d_decision") != "radio_process_model_revision_required":
        raise ValueError("Phase 3E requires the Phase 3D temporal-model rejection")
    if frozen.get("required_corrected_noise_decision") != "corrected_control_valid":
        raise ValueError("Phase 3E requires the corrected RFsim noise control")

    development = config.get("development") or {}
    if development.get("cross_validation") != "leave_one_complete_session_out":
        raise ValueError("Phase 3E requires leave-one-session-out development")
    if int(development.get("folds_required", 0)) != 5:
        raise ValueError("Phase 3E requires all five development folds")
    if development.get("final_test6_access") is not False:
        raise ValueError("Phase 3E must keep Test 6 inaccessible")

    candidates = config.get("candidates") or {}
    blocks = candidates.get("block_replays") or {}
    if blocks.get("lengths_rows") != [10, 20, 40, 60]:
        raise ValueError("unexpected Phase 3E block-length candidates")
    if not all(
        blocks.get(name) is True
        for name in (
            "circular_within_session",
            "paired_features",
            "preserve_session_boundaries_in_source_blocks",
        )
    ):
        raise ValueError("block replay must preserve pairs and source-session boundaries")
    var = candidates.get("empirical_innovation_var1") or {}
    if var.get("id") != "empirical_innovation_var1":
        raise ValueError("the empirical-innovation VAR(1) candidate is missing")
    switching = candidates.get("switching_var1") or {}
    if switching.get("id") != "student_t_2state_switching_var1":
        raise ValueError("the switching VAR(1) candidate is missing")
    for value in (var, switching):
        radius = float(value.get("maximum_spectral_radius", 0))
        if not 0 < radius < 1:
            raise ValueError("autoregressive spectral-radius bounds must be in (0, 1)")

    evaluation = config.get("evaluation") or {}
    if int(evaluation.get("generation_repetitions", 0)) < 20:
        raise ValueError("Phase 3E requires at least twenty generator repetitions")
    if evaluation.get("joint_distribution", {}).get("statistic") != (
        "biased_rbf_mmd_squared_v_statistic"
    ):
        raise ValueError("Phase 3E requires biased nonnegative MMD squared")
    if evaluation.get("reference") != "fold_training_real_session_pairwise_90th_percentile":
        raise ValueError("Phase 3E requires the frozen real-session reference")

    rules = config.get("decision_rules") or {}
    if int(rules.get("required_development_folds", 0)) != 5:
        raise ValueError("every development fold must determine the Phase 3E decision")
    for name in (
        "minimum_joint_supported_folds",
        "minimum_temporal_supported_folds",
        "minimum_joint_and_temporal_supported_folds",
    ):
        if int(rules.get(name, 0)) != 4:
            raise ValueError("Phase 3E support gates must require four of five folds")
    expected = [
        "paired_block_10",
        "paired_block_20",
        "paired_block_40",
        "paired_block_60",
        "empirical_innovation_var1",
        "student_t_2state_switching_var1",
    ]
    if rules.get("complexity_order") != expected:
        raise ValueError("unexpected Phase 3E candidate complexity order")

    claims = config.get("claim_limits") or {}
    for name in (
        "physical_channel_reconstruction",
        "absolute_rsrp_calibration",
        "absolute_noise_power_calibration",
        "final_test6_validation",
        "abc",
    ):
        if claims.get(name) != "prohibited":
            raise ValueError(f"Phase 3E claim limit must remain prohibited: {name}")
    reservation = config.get("reservation") or {}
    if reservation.get("request_now") is not False:
        raise ValueError("Phase 3E cannot request POWDER")
    if int(reservation.get("preparation_lead_time_minutes", 0)) < 30:
        raise ValueError("reservation notice must remain at least 30 minutes")


def _verify_inputs(
    *,
    archive: Path,
    phase3d_config: Path,
    phase3d_decision: Path,
    corrected_noise_result: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    observed = {
        "archive_sha256": _sha256(archive),
        "phase3d_config_sha256": _sha256(phase3d_config),
        "phase3d_decision_sha256": _sha256(phase3d_decision),
        "corrected_noise_result_sha256": _sha256(corrected_noise_result),
    }
    for name, checksum in observed.items():
        if checksum != frozen[name]:
            raise ValueError(f"Phase 3E frozen input checksum mismatch: {name}")
    phase3d = _read_json(phase3d_decision)
    if phase3d.get("decision_code") != frozen["required_phase3d_decision"]:
        raise ValueError("unexpected Phase 3D decision")
    if (phase3d.get("final_evaluation") or {}).get("payload_opened") is not False:
        raise ValueError("the Phase 3D final payload must remain unopened")
    corrected = _read_json(corrected_noise_result)
    if corrected.get("decision_code") != frozen["required_corrected_noise_decision"]:
        raise ValueError("unexpected corrected-noise result")
    if (corrected.get("development_comparison") or {}).get("final_test6_accessed") is not False:
        raise ValueError("the corrected-noise analysis must not access Test 6")
    return observed


def _implementation_paths() -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    return {
        "implementation": Path(__file__).resolve(),
        "tests": repository / "tests/test_upv_phase3e.py",
    }


def write_phase3e_protocol_freeze(
    *,
    config_path: str | Path,
    phase3d_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    config_file = Path(config_path).resolve()
    phase3d_file = Path(phase3d_config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3E protocol output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3e_config(config)
    phase3d_config = _read_yaml(phase3d_file)
    validate_phase3d_config(phase3d_config)
    if _sha256(phase3d_file) != config["frozen_inputs"]["phase3d_config_sha256"]:
        raise ValueError("Phase 3D configuration checksum mismatch")
    repository = _git_revision()
    if repository.get("tracked_worktree_dirty") is not False:
        raise ValueError("the Phase 3E protocol must be frozen from a clean tracked worktree")
    files = {
        "config": config_file,
        "phase3d_config": phase3d_file,
        **_implementation_paths(),
    }
    freeze = {
        "schema_version": 1,
        "stage": "phase_3e_protocol_freeze",
        "protocol_revision": config["protocol_revision"],
        "repository": repository,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in files.items()
        },
        "candidate_ids": config["decision_rules"]["complexity_order"],
        "development_folds": 5,
        "final_test6_access": False,
        "execution_authorized": False,
        "abc_authorized": False,
    }
    output.mkdir(parents=True)
    freeze_path = output / "protocol_freeze.json"
    _write_json(freeze_path, freeze)
    checksum_path = output / "SHA256SUMS.json"
    _write_json(checksum_path, {freeze_path.name: _sha256(freeze_path)})
    return {"protocol_freeze": str(freeze_path), "checksums": str(checksum_path)}


def _verify_protocol_freeze(
    protocol_dir: Path,
    config_path: Path,
    phase3d_config_path: Path,
) -> dict[str, Any]:
    freeze_path = protocol_dir / "protocol_freeze.json"
    checksums_path = protocol_dir / "SHA256SUMS.json"
    if not freeze_path.is_file() or not checksums_path.is_file():
        raise ValueError("the Phase 3E protocol freeze is incomplete")
    checksums = _read_json(checksums_path)
    if checksums.get(freeze_path.name) != _sha256(freeze_path):
        raise ValueError("Phase 3E protocol-freeze checksum mismatch")
    freeze = _read_json(freeze_path)
    if freeze.get("stage") != "phase_3e_protocol_freeze":
        raise ValueError("unexpected Phase 3E freeze stage")
    if freeze.get("final_test6_access") is not False:
        raise ValueError("the Phase 3E freeze must keep Test 6 inaccessible")
    observed_paths = {
        "config": config_path,
        "phase3d_config": phase3d_config_path,
        **_implementation_paths(),
    }
    inputs = freeze.get("inputs") or {}
    for name, path in observed_paths.items():
        if (inputs.get(name) or {}).get("sha256") != _sha256(path):
            raise ValueError(f"Phase 3E protocol input changed after freeze: {name}")
    return freeze


def _stabilize(matrix: np.ndarray, maximum_radius: float) -> tuple[np.ndarray, float, float]:
    eigenvalues = np.linalg.eigvals(matrix)
    observed = float(np.max(np.abs(eigenvalues)))
    result = np.asarray(matrix, dtype=float)
    if observed > maximum_radius:
        result = result * (maximum_radius / observed)
    stabilized = float(np.max(np.abs(np.linalg.eigvals(result))))
    return result, observed, stabilized


def _fit_weighted_var1(
    previous: np.ndarray,
    current: np.ndarray,
    weights: np.ndarray,
    regularization: float,
    maximum_radius: float,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    if len(previous) != len(current) or len(previous) != len(weights):
        raise ValueError("weighted VAR(1) arrays differ in length")
    effective = float(weights.sum())
    if effective <= 2.0:
        raise ValueError("insufficient effective transitions for VAR(1)")
    design = np.column_stack([np.ones(len(previous)), previous])
    normal = design.T @ (weights[:, None] * design)
    penalty = np.eye(normal.shape[0]) * regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        normal + penalty,
        design.T @ (weights[:, None] * current),
    )
    intercept = coefficients[0]
    matrix, raw_radius, stabilized_radius = _stabilize(coefficients[1:].T, maximum_radius)
    residuals = current - (intercept + previous @ matrix.T)
    probabilities = weights / weights.sum()
    return {
        "intercept": intercept,
        "matrix": matrix,
        "residuals": residuals,
        "residual_probabilities": probabilities,
        "effective_transitions": effective,
        "raw_spectral_radius": raw_radius,
        "spectral_radius": stabilized_radius,
    }


def _transition_rows(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    previous = np.vstack([sequence[:-1] for sequence in sequences if len(sequence) > 1])
    current = np.vstack([sequence[1:] for sequence in sequences if len(sequence) > 1])
    return previous, current


def _fit_var1(sequences: list[np.ndarray], config: dict[str, Any]) -> dict[str, Any]:
    candidate = config["candidates"]["empirical_innovation_var1"]
    previous, current = _transition_rows(sequences)
    fitted = _fit_weighted_var1(
        previous,
        current,
        np.ones(len(previous)),
        float(candidate["covariance_regularization"]),
        float(candidate["maximum_spectral_radius"]),
    )
    return {
        "candidate_id": candidate["id"],
        "initial_values": np.vstack(sequences),
        **fitted,
        "eligible": bool(np.isfinite(fitted["spectral_radius"])),
    }


def _sample_var1(
    model: dict[str, Any], lengths: list[int], rng: np.random.Generator
) -> list[np.ndarray]:
    initial = np.asarray(model["initial_values"], dtype=float)
    residuals = np.asarray(model["residuals"], dtype=float)
    probabilities = np.asarray(model["residual_probabilities"], dtype=float)
    matrix = np.asarray(model["matrix"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    generated = []
    for length in lengths:
        values = np.empty((length, 2), dtype=float)
        values[0] = initial[int(rng.integers(0, len(initial)))]
        for index in range(1, length):
            innovation = residuals[int(rng.choice(len(residuals), p=probabilities))]
            values[index] = intercept + matrix @ values[index - 1] + innovation
        generated.append(values)
    return generated


def _fit_switching_var1(
    sequences: list[np.ndarray], phase3d_config: dict[str, Any], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    candidate_config = config["candidates"]["switching_var1"]
    hmm_candidate = next(
        value
        for value in phase3d_config["models"]["candidates"]
        if value["id"] == "student_t_2state"
    )
    hmm = _fit_hmm(sequences, hmm_candidate, phase3d_config, seed)
    previous_rows = []
    current_rows = []
    gamma_rows = []
    initial_rows = []
    initial_gamma = []
    for sequence in sequences:
        _, gamma, _ = _sequence_posteriors(sequence, hmm)
        if len(sequence) > 1:
            previous_rows.append(sequence[:-1])
            current_rows.append(sequence[1:])
            gamma_rows.append(gamma[1:])
        initial_rows.append(sequence)
        initial_gamma.append(gamma)
    previous = np.vstack(previous_rows)
    current = np.vstack(current_rows)
    gammas = np.vstack(gamma_rows)
    all_values = np.vstack(initial_rows)
    all_gamma = np.vstack(initial_gamma)
    state_models = []
    minimum_effective = float(candidate_config["minimum_effective_transitions_per_state"])
    for state in range(2):
        fitted = _fit_weighted_var1(
            previous,
            current,
            gammas[:, state],
            float(candidate_config["covariance_regularization"]),
            float(candidate_config["maximum_spectral_radius"]),
        )
        initial_weights = all_gamma[:, state]
        initial_weights = initial_weights / initial_weights.sum()
        fitted["initial_values"] = all_values
        fitted["initial_probabilities"] = initial_weights
        fitted["eligible"] = fitted["effective_transitions"] >= minimum_effective
        state_models.append(fitted)
    return {
        "candidate_id": candidate_config["id"],
        "initial": np.asarray(hmm["initial"], dtype=float),
        "transition": np.asarray(hmm["transition"], dtype=float),
        "states": state_models,
        "base_hmm": hmm,
        "eligible": bool(hmm["eligible"] and all(value["eligible"] for value in state_models)),
    }


def _sample_switching_var1(
    model: dict[str, Any], lengths: list[int], rng: np.random.Generator
) -> list[np.ndarray]:
    transition = np.asarray(model["transition"], dtype=float)
    initial = np.asarray(model["initial"], dtype=float)
    generated = []
    for length in lengths:
        values = np.empty((length, 2), dtype=float)
        state = int(rng.choice(2, p=initial))
        state_model = model["states"][state]
        initial_values = np.asarray(state_model["initial_values"], dtype=float)
        initial_probabilities = np.asarray(state_model["initial_probabilities"], dtype=float)
        values[0] = initial_values[int(rng.choice(len(initial_values), p=initial_probabilities))]
        for index in range(1, length):
            state = int(rng.choice(2, p=transition[state]))
            state_model = model["states"][state]
            residuals = np.asarray(state_model["residuals"], dtype=float)
            residual_probabilities = np.asarray(state_model["residual_probabilities"], dtype=float)
            innovation = residuals[int(rng.choice(len(residuals), p=residual_probabilities))]
            values[index] = (
                np.asarray(state_model["intercept"], dtype=float)
                + np.asarray(state_model["matrix"], dtype=float) @ values[index - 1]
                + innovation
            )
        generated.append(values)
    return generated


def _var_record(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": model["candidate_id"],
        "intercept": np.asarray(model["intercept"]).tolist(),
        "matrix": np.asarray(model["matrix"]).tolist(),
        "innovation_rows": len(model["residuals"]),
        "effective_transitions": float(model["effective_transitions"]),
        "raw_spectral_radius": float(model["raw_spectral_radius"]),
        "spectral_radius": float(model["spectral_radius"]),
        "eligible": bool(model["eligible"]),
    }


def _switching_record(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": model["candidate_id"],
        "initial": np.asarray(model["initial"]).tolist(),
        "transition": np.asarray(model["transition"]).tolist(),
        "states": [
            {
                "intercept": np.asarray(value["intercept"]).tolist(),
                "matrix": np.asarray(value["matrix"]).tolist(),
                "innovation_rows": len(value["residuals"]),
                "effective_transitions": float(value["effective_transitions"]),
                "raw_spectral_radius": float(value["raw_spectral_radius"]),
                "spectral_radius": float(value["spectral_radius"]),
                "eligible": bool(value["eligible"]),
            }
            for value in model["states"]
        ],
        "base_hmm": _model_record(model["base_hmm"]),
        "eligible": bool(model["eligible"]),
    }


def _decision(results: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    rules = config["decision_rules"]
    required = int(rules["required_development_folds"])
    complexity = {name: index for index, name in enumerate(rules["complexity_order"])}
    candidates = []
    passing = []
    for candidate_id in rules["complexity_order"]:
        frame = results.loc[results["candidate_id"].eq(candidate_id)].copy()
        if len(frame) != required:
            raise ValueError(f"incomplete Phase 3E candidate: {candidate_id}")
        joint = frame["joint_mmd_squared_mean"] <= frame["joint_reference_p90"]
        temporal = frame["temporal_error_mean"] <= frame["temporal_reference_p90"]
        both = joint & temporal
        normalized = (
            frame["joint_mmd_squared_mean"] / frame["joint_reference_p90"]
            + frame["temporal_error_mean"] / frame["temporal_reference_p90"]
        )
        record = {
            "candidate_id": candidate_id,
            "joint_supported_folds": int(joint.sum()),
            "temporal_supported_folds": int(temporal.sum()),
            "joint_and_temporal_supported_folds": int(both.sum()),
            "median_reference_normalized_error_sum": float(np.median(normalized)),
            "eligible_in_every_fold": bool(frame["model_eligible"].all()),
            "complexity_rank": complexity[candidate_id],
        }
        record["passes"] = bool(
            record["eligible_in_every_fold"]
            and record["joint_supported_folds"] >= int(rules["minimum_joint_supported_folds"])
            and record["temporal_supported_folds"] >= int(rules["minimum_temporal_supported_folds"])
            and record["joint_and_temporal_supported_folds"]
            >= int(rules["minimum_joint_and_temporal_supported_folds"])
        )
        candidates.append(record)
        if record["passes"]:
            passing.append(record)
    if passing:
        selected = sorted(
            passing,
            key=lambda value: (
                -value["joint_and_temporal_supported_folds"],
                value["median_reference_normalized_error_sum"],
                value["complexity_rank"],
            ),
        )[0]
        outcome = rules["outcomes"]["supported"]
        selected_process = selected["candidate_id"]
    else:
        outcome = rules["outcomes"]["unsupported"]
        selected_process = None
    return {
        "decision_code": outcome["code"],
        "selected_process": selected_process,
        "next_action": outcome["next_action"],
        "candidate_support": candidates,
        "powder_reservation_authorized": False,
        "final_evaluation_authorized": False,
        "abc_authorized": False,
    }


def _fit_selected_on_all_development(
    *,
    selected: str,
    sessions: dict[str, pd.DataFrame],
    phase3d_config: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    route_means = _fit_route_means(sessions, phase3d_config)
    detrended = []
    for frame in sessions.values():
        transformed, audit = _apply_route_means(frame, route_means, phase3d_config)
        if audit["unsupported_fraction"] > 0:
            raise ValueError("all-development route fit unexpectedly lacks support")
        detrended.extend(_feature_sequences(transformed, ["rsrp_residual_db", "sinr_residual_db"]))
    if selected.startswith("paired_block_"):
        record = {
            "candidate_id": selected,
            "block_length_rows": int(selected.rsplit("_", 1)[1]),
            "source_sequence_count": len(detrended),
            "source_rows": int(sum(len(value) for value in detrended)),
            "refit_required_for_replay": True,
        }
    elif selected == "empirical_innovation_var1":
        record = _var_record(_fit_var1(detrended, config))
        record["refit_required_for_replay"] = True
    elif selected == "student_t_2state_switching_var1":
        record = _switching_record(_fit_switching_var1(detrended, phase3d_config, config, seed))
        record["refit_required_for_replay"] = True
    else:
        raise ValueError(f"unexpected selected process: {selected}")
    return route_means, record


def analyze_phase3e_radio_process(
    *,
    archive_path: str | Path,
    phase3d_config_path: str | Path,
    phase3d_decision_path: str | Path,
    corrected_noise_result_path: str | Path,
    protocol_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    archive = Path(archive_path).resolve()
    phase3d_config_file = Path(phase3d_config_path).resolve()
    phase3d_decision = Path(phase3d_decision_path).resolve()
    corrected_noise = Path(corrected_noise_result_path).resolve()
    protocol = Path(protocol_dir).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Phase 3E output already exists: {output}")
    config = _read_yaml(config_file)
    validate_phase3e_config(config)
    phase3d_config = _read_yaml(phase3d_config_file)
    validate_phase3d_config(phase3d_config)
    input_checksums = _verify_inputs(
        archive=archive,
        phase3d_config=phase3d_config_file,
        phase3d_decision=phase3d_decision,
        corrected_noise_result=corrected_noise,
        config=config,
    )
    freeze = _verify_protocol_freeze(protocol, config_file, phase3d_config_file)
    sessions, quality = _load_development_sessions(archive, phase3d_config)
    if phase3d_config["final_evaluation"]["source_path"] in set(quality["source_path"]):
        raise AssertionError("the final evaluation payload entered Phase 3E development")

    repetitions = int(config["evaluation"]["generation_repetitions"])
    seed = int(config["evaluation"]["seed"])
    maximum_rows = int(config["evaluation"]["joint_distribution"]["maximum_rows_per_trace"])
    lags = [
        int(value) for value in config["evaluation"]["temporal"]["autocorrelation_lags_seconds"]
    ]
    maximum_unsupported = float(
        phase3d_config["decision_rules"]["maximum_unsupported_position_fraction_per_fold"]
    )
    fold_rows = []
    model_records: dict[str, Any] = {}
    route_rows = []
    session_ids = sorted(sessions)
    candidate_ids = config["decision_rules"]["complexity_order"]
    for fold_index, holdout_id in enumerate(session_ids, start=1):
        fold_id = f"fold_{fold_index}_{holdout_id}"
        training = {key: value for key, value in sessions.items() if key != holdout_id}
        route_means = _fit_route_means(training, phase3d_config)
        route_means["fold_id"] = fold_id
        route_rows.append(route_means)
        residual_training = []
        for session_id, frame in training.items():
            transformed, audit = _apply_route_means(frame, route_means, phase3d_config)
            if audit["unsupported_fraction"] > maximum_unsupported:
                raise ValueError(f"training spatial support failed: {fold_id}/{session_id}")
            residual_training.extend(
                _feature_sequences(transformed, ["rsrp_residual_db", "sinr_residual_db"])
            )
        holdout, support = _apply_route_means(sessions[holdout_id], route_means, phase3d_config)
        if support["unsupported_fraction"] > maximum_unsupported:
            raise ValueError(f"holdout spatial support failed: {fold_id}")
        observed = _feature_sequences(holdout, ["relative_rsrp_db", "sinr_db"])
        route_sequences = _feature_sequences(holdout, ["route_relative_rsrp_db", "route_sinr_db"])
        center, scale = _balanced_scale(training)
        training_targets = np.vstack(
            [frame[["relative_rsrp_db", "sinr_db"]].to_numpy(float) for frame in training.values()]
        )
        bandwidth = median_heuristic_bandwidth((training_targets - center) / scale)
        joint_reference, temporal_reference = _real_pairwise_reference(
            training, center, scale, bandwidth, maximum_rows, lags
        )

        generators: dict[str, tuple[Any, bool]] = {}
        for block_length in config["candidates"]["block_replays"]["lengths_rows"]:
            candidate_id = f"paired_block_{int(block_length)}"
            generators[candidate_id] = (
                lambda lengths, rng, block=int(block_length), source=residual_training: (
                    _sample_blocks(source, lengths, block, rng)
                ),
                True,
            )
        var_model = _fit_var1(residual_training, config)
        generators["empirical_innovation_var1"] = (
            lambda lengths, rng, model=var_model: _sample_var1(model, lengths, rng),
            bool(var_model["eligible"]),
        )
        switching_model = _fit_switching_var1(
            residual_training,
            phase3d_config,
            config,
            seed + fold_index * 1000000,
        )
        generators["student_t_2state_switching_var1"] = (
            lambda lengths, rng, model=switching_model: _sample_switching_var1(model, lengths, rng),
            bool(switching_model["eligible"]),
        )
        model_records[fold_id] = {
            "empirical_innovation_var1": _var_record(var_model),
            "student_t_2state_switching_var1": _switching_record(switching_model),
        }
        for candidate_index, candidate_id in enumerate(candidate_ids, start=1):
            generator, eligible = generators[candidate_id]
            metrics = _evaluate_generator(
                observed=observed,
                route_sequences=route_sequences,
                generator=generator,
                repetitions=repetitions,
                seed=seed + fold_index * 100000 + candidate_index * 1000,
                center=center,
                scale=scale,
                bandwidth=bandwidth,
                maximum_rows=maximum_rows,
                lags=lags,
            )
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "holdout_session_id": holdout_id,
                    "candidate_id": candidate_id,
                    "model_eligible": eligible,
                    "supported_position_fraction": 1.0 - support["unsupported_fraction"],
                    "joint_reference_p90": joint_reference,
                    "temporal_reference_p90": temporal_reference,
                    **metrics,
                }
            )

    results = pd.DataFrame(fold_rows)
    decision = _decision(results, config)
    selected_route_means = pd.DataFrame()
    selected_record: dict[str, Any] | None = None
    if decision["selected_process"] is not None:
        selected_route_means, selected_record = _fit_selected_on_all_development(
            selected=decision["selected_process"],
            sessions=sessions,
            phase3d_config=phase3d_config,
            config=config,
            seed=seed + 9000000,
        )
    result = {
        "schema_version": 1,
        "stage": "phase_3e_offline_temporal_process_result",
        "research_question": config["research_question"],
        "protocol_revision": config["protocol_revision"],
        "protocol_repository": freeze["repository"],
        "analysis_repository": _git_revision(),
        "input_sha256": input_checksums,
        "development_sessions": session_ids,
        "development_folds": len(session_ids),
        "final_evaluation": {
            "source_path": phase3d_config["final_evaluation"]["source_path"],
            "payload_opened": False,
            "authorized": False,
        },
        **decision,
        "reservation": config["reservation"],
        "claim_limits": config["claim_limits"],
    }
    output.mkdir(parents=True)
    paths = {
        "phase3e_decision.json": result,
        "fitted_models.json": model_records,
        "analysis_manifest.json": {
            "schema_version": 1,
            "stage": "phase_3e_analysis_manifest",
            "archive_sha256": input_checksums["archive_sha256"],
            "protocol_freeze_sha256": _sha256(protocol / "protocol_freeze.json"),
            "folds": len(session_ids),
            "candidates": candidate_ids,
            "generation_repetitions_per_candidate_fold": repetitions,
            "final_evaluation_payload_opened": False,
            "reservation_requested": False,
        },
    }
    for name, value in paths.items():
        _write_json(output / name, value)
    _write_csv(output / "fold_candidate_metrics.csv", results)
    _write_csv(output / "fold_route_means.csv", pd.concat(route_rows, ignore_index=True))
    _write_csv(output / "development_input_quality.csv", quality)
    if selected_record is not None:
        _write_json(
            output / "selected_process.json",
            {
                "schema_version": 1,
                "stage": "phase_3e_selected_development_process",
                "candidate_id": decision["selected_process"],
                "model": selected_record,
                "final_test6_accessed": False,
                "hardware_execution_authorized": False,
            },
        )
        _write_csv(output / "selected_route_means.csv", selected_route_means)
    checksums = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(output / "SHA256SUMS.json", checksums)
    return {
        "output": str(output),
        "decision": decision["decision_code"],
        "selected_process": decision["selected_process"],
        "final_test6_accessed": False,
        "reservation_requested": False,
    }
