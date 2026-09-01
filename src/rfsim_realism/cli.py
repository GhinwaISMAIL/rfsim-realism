from __future__ import annotations

import argparse
import json

from .distribution import run_distribution_analysis
from .distribution_calibration import run_distribution_calibration
from .family_compare import run_family_comparison
from .fetch import fetch_archive
from .mapping import run_static_mapping
from .mmd_abc import run_mmd_abc, write_mmd_abc_plan, write_posterior_predictive_plan
from .noise_response import write_corrected_noise_response_evaluation
from .report import build_report
from .static_grid import plan_document as static_grid_plan
from .static_grid import run_campaign as run_static_grid
from .sweep import load_config, plan_document, run_campaign, write_json
from .ucc_static import build_manifest, write_manifest
from .upv_measurement_audit import run_measurement_equivalence_audit
from .upv_phase3b import analyze_phase3b_support
from .upv_phase3c import build_phase3c_plan, write_deterministic_replay_evaluation
from .upv_phase3c2 import run_phase3c2_trace_validation
from .upv_phase3c13 import write_static_tdlb_pilot_evaluation
from .upv_phase3c14 import write_awgn_execution_control_evaluation
from .upv_phase3c15 import write_phase3c15_support_analysis
from .upv_phase3d import analyze_phase3d_radio_process, write_phase3d_protocol_freeze
from .upv_phase3e import analyze_phase3e_radio_process, write_phase3e_protocol_freeze
from .upv_phase3f import analyze_phase3f_exchangeability
from .upv_phase3g import prepare_phase3g_direct_trace
from .upv_phase3g_diagnosis import diagnose_phase3g_boundary
from .upv_phase3g_response import analyze_phase3g_bounded_response
from .upv_phase3h import analyze_phase3h_dynamic_staircase, freeze_phase3h_dynamic_staircase
from .upv_phase3i import (
    analyze_phase3i_short_trace,
    freeze_phase3i_short_trace,
    recover_phase3i_short_trace,
)
from .upv_phase3j import freeze_phase3j_full_trace
from .upv_protocol import prepare_upv_protocol
from .upv_support import analyze_upv_support
from .upv_support_v2 import write_upv_support_v2_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rfsim-realism")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch-ucc")
    fetch.add_argument("--output", required=True)

    curate = commands.add_parser("curate-static")
    curate.add_argument("--dataset", required=True)
    curate.add_argument("--output", required=True)

    report = commands.add_parser("static-report")
    report.add_argument("--manifest", required=True)
    report.add_argument("--output", required=True)

    plan = commands.add_parser("sweep-plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)

    run = commands.add_parser("sweep-run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--dashboard-repo", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--point", action="append")
    run.add_argument("--limit", type=int)

    grid_plan = commands.add_parser("grid-plan")
    grid_plan.add_argument("--config", required=True)
    grid_plan.add_argument("--output", required=True)

    grid_run = commands.add_parser("grid-run")
    grid_run.add_argument("--config", required=True)
    grid_run.add_argument("--run-dir", required=True)
    grid_run.add_argument("--dashboard-repo", required=True)
    grid_run.add_argument("--state", required=True)
    grid_run.add_argument("--point", action="append")
    grid_run.add_argument("--limit", type=int)

    mapping = commands.add_parser("static-map")
    mapping.add_argument("--dataset-dir", required=True)
    mapping.add_argument("--selection-manifest", required=True)
    mapping.add_argument("--campaign-state", required=True)
    mapping.add_argument("--ucc-manifest", required=True)
    mapping.add_argument("--comparison-contract", required=True)
    mapping.add_argument("--config", required=True)
    mapping.add_argument("--output", required=True)

    distribution = commands.add_parser("rf-distribution")
    distribution.add_argument("--dataset", required=True)
    distribution.add_argument("--manifest", required=True)
    distribution.add_argument("--mapping-dir")
    distribution.add_argument("--config", required=True)
    distribution.add_argument("--output", required=True)

    distribution_calibration = commands.add_parser("distribution-calibrate")
    distribution_calibration.add_argument("--real-observations", required=True)
    distribution_calibration.add_argument("--executions-root", required=True)
    distribution_calibration.add_argument("--selection-manifest", required=True)
    distribution_calibration.add_argument("--campaign-state", required=True)
    distribution_calibration.add_argument("--config", required=True)
    distribution_calibration.add_argument("--output", required=True)

    mmd_abc_plan = commands.add_parser("mmd-abc-plan")
    mmd_abc_plan.add_argument("--config", required=True)
    mmd_abc_plan.add_argument("--output", required=True)

    mmd_abc = commands.add_parser("mmd-abc-infer")
    mmd_abc.add_argument("--real-observations", required=True)
    mmd_abc.add_argument("--executions-root", required=True)
    mmd_abc.add_argument("--proposal-plan", required=True)
    mmd_abc.add_argument("--campaign-state", required=True)
    mmd_abc.add_argument("--config", required=True)
    mmd_abc.add_argument("--output", required=True)

    validation_plan = commands.add_parser("mmd-abc-validation-plan")
    validation_plan.add_argument("--calibration-dir", required=True)
    validation_plan.add_argument("--config", required=True)
    validation_plan.add_argument("--output", required=True)

    family_compare = commands.add_parser("family-compare")
    family_compare.add_argument("--primary-dir", required=True)
    family_compare.add_argument("--candidate-dir", required=True)
    family_compare.add_argument("--primary-label", required=True)
    family_compare.add_argument("--candidate-label", required=True)
    family_compare.add_argument("--output", required=True)

    upv = commands.add_parser("prepare-upv")
    upv.add_argument("--archive", required=True)
    upv.add_argument("--config", required=True)
    upv.add_argument("--output", required=True)

    upv_support = commands.add_parser("analyze-upv-support")
    upv_support.add_argument("--route-observations", required=True)
    upv_support.add_argument("--locked-split", required=True)
    upv_support.add_argument("--upv-archive", required=True)
    upv_support.add_argument("--phase1-config", required=True)
    upv_support.add_argument("--selection-manifest", required=True)
    upv_support.add_argument("--campaign-state", required=True)
    upv_support.add_argument("--executions-root", required=True)
    upv_support.add_argument("--config", required=True)
    upv_support.add_argument("--output", required=True)

    upv_audit = commands.add_parser("audit-upv-measurement")
    upv_audit.add_argument("--upv-archive", required=True)
    upv_audit.add_argument("--phase2-manifest", required=True)
    upv_audit.add_argument("--phase2-gate", required=True)
    upv_audit.add_argument("--oai-source", required=True)
    upv_audit.add_argument("--profile-source", required=True)
    upv_audit.add_argument("--config", required=True)
    upv_audit.add_argument("--output", required=True)

    upv_v2 = commands.add_parser("plan-upv-support-v2")
    upv_v2.add_argument("--phase3a-decision", required=True)
    upv_v2.add_argument("--phase3a-gate", required=True)
    upv_v2.add_argument("--config", required=True)
    upv_v2.add_argument("--output", required=True)

    upv_phase3b = commands.add_parser("analyze-upv-phase3b")
    upv_phase3b.add_argument("--route-observations", required=True)
    upv_phase3b.add_argument("--locked-split", required=True)
    upv_phase3b.add_argument("--upv-archive", required=True)
    upv_phase3b.add_argument("--phase1-config", required=True)
    upv_phase3b.add_argument("--selection-manifest", required=True)
    upv_phase3b.add_argument("--campaign-state", required=True)
    upv_phase3b.add_argument("--executions-root", required=True)
    upv_phase3b.add_argument("--phase3a-decision", required=True)
    upv_phase3b.add_argument("--phase3a-gate", required=True)
    upv_phase3b.add_argument("--public-evidence", required=True)
    upv_phase3b.add_argument("--config", required=True)
    upv_phase3b.add_argument("--output", required=True)

    upv_phase3c = commands.add_parser("plan-upv-phase3c")
    upv_phase3c.add_argument("--phase3b-decision", required=True)
    upv_phase3c.add_argument("--phase3b-gate", required=True)
    upv_phase3c.add_argument("--oai-source", required=True)
    upv_phase3c.add_argument("--config", required=True)
    upv_phase3c.add_argument("--output", required=True)

    phase3c_replay = commands.add_parser("evaluate-upv-phase3c-replay")
    phase3c_replay.add_argument("--telemetry", required=True)
    phase3c_replay.add_argument("--plan-dir", required=True)
    phase3c_replay.add_argument("--output", required=True)

    phase3c2 = commands.add_parser("validate-upv-phase3c2-trace")
    phase3c2.add_argument("--config", required=True)
    phase3c2.add_argument("--phase3c1-result", required=True)
    phase3c2.add_argument("--sample-rate-evidence", required=True)
    phase3c2.add_argument("--oai-source", required=True)
    phase3c2.add_argument("--output", required=True)
    phase3c2.add_argument("--manifest-output", required=True)

    phase3c13 = commands.add_parser("evaluate-upv-phase3c13-static-tdlb")
    phase3c13.add_argument("--telemetry", required=True)
    phase3c13.add_argument("--execution-state", required=True)
    phase3c13.add_argument("--config", required=True)
    phase3c13.add_argument("--identity-amendment")
    phase3c13.add_argument("--output", required=True)

    phase3c14 = commands.add_parser("evaluate-upv-phase3c14-awgn-control")
    phase3c14.add_argument("--telemetry", required=True)
    phase3c14.add_argument("--execution-state", required=True)
    phase3c14.add_argument("--config", required=True)
    phase3c14.add_argument("--tdlb-evaluation", required=True)
    phase3c14.add_argument("--tdlb-result", required=True)
    phase3c14.add_argument("--identity-amendment")
    phase3c14.add_argument("--output", required=True)

    phase3c15 = commands.add_parser("analyze-upv-phase3c15-support")
    phase3c15.add_argument("--route-observations", required=True)
    phase3c15.add_argument("--locked-spatial-split", required=True)
    phase3c15.add_argument("--phase3c14-telemetry", required=True)
    phase3c15.add_argument("--phase3c14-evaluation", required=True)
    phase3c15.add_argument("--phase3c14-result", required=True)
    phase3c15.add_argument("--phase3b-decision", required=True)
    phase3c15.add_argument("--phase3b-distribution-diagnostics", required=True)
    phase3c15.add_argument("--phase3b-locked-validation-support", required=True)
    phase3c15.add_argument("--config", required=True)
    phase3c15.add_argument("--output", required=True)

    phase3d_freeze = commands.add_parser("freeze-upv-phase3d-radio-process")
    phase3d_freeze.add_argument("--archive", required=True)
    phase3d_freeze.add_argument("--phase3c15-result", required=True)
    phase3d_freeze.add_argument("--config", required=True)
    phase3d_freeze.add_argument("--output", required=True)

    phase3d = commands.add_parser("analyze-upv-phase3d-radio-process")
    phase3d.add_argument("--archive", required=True)
    phase3d.add_argument("--phase3c15-result", required=True)
    phase3d.add_argument("--protocol-dir", required=True)
    phase3d.add_argument("--config", required=True)
    phase3d.add_argument("--output", required=True)

    noise_response = commands.add_parser("evaluate-corrected-rfsim-noise-response")
    noise_response.add_argument("--raw-archive", required=True)
    noise_response.add_argument("--telemetry", required=True)
    noise_response.add_argument("--execution-state", required=True)
    noise_response.add_argument("--protocol", required=True)
    noise_response.add_argument("--hardware-freeze", required=True)
    noise_response.add_argument("--analysis-spec", required=True)
    noise_response.add_argument("--development-route-means", required=True)
    noise_response.add_argument("--phase3d-decision", required=True)
    noise_response.add_argument("--output", required=True)

    phase3e_freeze = commands.add_parser("freeze-upv-phase3e-radio-process")
    phase3e_freeze.add_argument("--config", required=True)
    phase3e_freeze.add_argument("--phase3d-config", required=True)
    phase3e_freeze.add_argument("--output", required=True)

    phase3e = commands.add_parser("analyze-upv-phase3e-radio-process")
    phase3e.add_argument("--archive", required=True)
    phase3e.add_argument("--phase3d-config", required=True)
    phase3e.add_argument("--phase3d-decision", required=True)
    phase3e.add_argument("--corrected-noise-result", required=True)
    phase3e.add_argument("--protocol-dir", required=True)
    phase3e.add_argument("--config", required=True)
    phase3e.add_argument("--output", required=True)

    phase3f = commands.add_parser("analyze-upv-phase3f-exchangeability")
    phase3f.add_argument("--archive", required=True)
    phase3f.add_argument("--phase3d-config", required=True)
    phase3f.add_argument("--phase3e-result", required=True)
    phase3f.add_argument("--config", required=True)
    phase3f.add_argument("--output", required=True)

    phase3g = commands.add_parser("prepare-upv-phase3g-direct-trace")
    phase3g.add_argument("--archive", required=True)
    phase3g.add_argument("--phase3d-config", required=True)
    phase3g.add_argument("--phase3f-result", required=True)
    phase3g.add_argument("--scalar-control-result", required=True)
    phase3g.add_argument("--corrected-noise-result", required=True)
    phase3g.add_argument("--config", required=True)
    phase3g.add_argument("--output", required=True)

    phase3g_response = commands.add_parser("analyze-upv-phase3g-bounded-response")
    phase3g_response.add_argument("--campaign-dir", required=True)
    phase3g_response.add_argument("--archive", required=True)
    phase3g_response.add_argument("--direct-config", required=True)
    phase3g_response.add_argument("--execution-config", required=True)
    phase3g_response.add_argument("--output", required=True)

    phase3g_diagnosis = commands.add_parser("diagnose-upv-phase3g-boundary")
    phase3g_diagnosis.add_argument("--response-dir", required=True)
    phase3g_diagnosis.add_argument("--direct-trace", required=True)
    phase3g_diagnosis.add_argument("--output", required=True)

    phase3h_freeze = commands.add_parser("freeze-upv-phase3h-dynamic-staircase")
    phase3h_freeze.add_argument("--config", required=True)
    phase3h_freeze.add_argument("--diagnosis", required=True)
    phase3h_freeze.add_argument("--execution-medians", required=True)
    phase3h_freeze.add_argument("--direct-trace", required=True)
    phase3h_freeze.add_argument("--output", required=True)

    phase3h = commands.add_parser("analyze-upv-phase3h-dynamic-staircase")
    phase3h.add_argument("--campaign-dir", required=True)
    phase3h.add_argument("--protocol-dir", required=True)
    phase3h.add_argument("--config", required=True)
    phase3h.add_argument("--output", required=True)

    phase3i_freeze = commands.add_parser("freeze-upv-phase3i-short-trace")
    phase3i_freeze.add_argument("--config", required=True)
    phase3i_freeze.add_argument("--phase3h-decision", required=True)
    phase3i_freeze.add_argument("--phase3g-execution-medians", required=True)
    phase3i_freeze.add_argument("--phase3h-state-validation", required=True)
    phase3i_freeze.add_argument("--direct-trace", required=True)
    phase3i_freeze.add_argument("--output", required=True)

    phase3i_recover = commands.add_parser("recover-upv-phase3i-short-trace")
    phase3i_recover.add_argument("--campaign-dir", required=True)
    phase3i_recover.add_argument("--config", required=True)
    phase3i_recover.add_argument("--output", required=True)

    phase3i = commands.add_parser("analyze-upv-phase3i-short-trace")
    phase3i.add_argument("--campaign-dir", required=True)
    phase3i.add_argument("--protocol-dir", required=True)
    phase3i.add_argument("--config", required=True)
    phase3i.add_argument("--output", required=True)

    phase3j_freeze = commands.add_parser("freeze-upv-phase3j-full-trace")
    phase3j_freeze.add_argument("--config", required=True)
    phase3j_freeze.add_argument("--phase3i-decision", required=True)
    phase3j_freeze.add_argument("--phase3g-execution-medians", required=True)
    phase3j_freeze.add_argument("--phase3h-state-validation", required=True)
    phase3j_freeze.add_argument("--direct-trace", required=True)
    phase3j_freeze.add_argument("--pyproject", required=True)
    phase3j_freeze.add_argument("--uv-lock", required=True)
    phase3j_freeze.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fetch-ucc":
        output = fetch_archive(args.output)
        print(json.dumps({"archive": str(output), "verified": True}, sort_keys=True))
        return
    if args.command == "curate-static":
        manifest = build_manifest(args.dataset)
        output = write_manifest(manifest, args.output)
        print(json.dumps({"manifest": str(output), **manifest["inventory"]}, sort_keys=True))
        return
    if args.command == "sweep-plan":
        output = write_json(args.output, plan_document(load_config(args.config)))
        print(json.dumps({"plan": str(output)}, sort_keys=True))
        return
    if args.command == "sweep-run":
        state = run_campaign(
            args.config,
            args.run_dir,
            args.dashboard_repo,
            args.state,
            point_ids=set(args.point) if args.point else None,
            limit=args.limit,
        )
        print(json.dumps({
            "state": args.state,
            "completed": len(state["completed"]),
            "failures": len(state["failures"]),
        }, sort_keys=True))
        return
    if args.command == "grid-plan":
        output = write_json(
            args.output, static_grid_plan(load_config(args.config)))
        print(json.dumps({"plan": str(output)}, sort_keys=True))
        return
    if args.command == "grid-run":
        state = run_static_grid(
            args.config,
            args.run_dir,
            args.dashboard_repo,
            args.state,
            point_ids=set(args.point) if args.point else None,
            limit=args.limit,
        )
        print(json.dumps({
            "state": args.state,
            "completed": len(state["completed"]),
            "failures": len(state["failures"]),
        }, sort_keys=True))
        return
    if args.command == "static-map":
        result = run_static_mapping(
            dataset_dir=args.dataset_dir,
            selection_manifest=args.selection_manifest,
            campaign_state=args.campaign_state,
            ucc_manifest=args.ucc_manifest,
            comparison_contract=args.comparison_contract,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "rf-distribution":
        result = run_distribution_analysis(
            dataset=args.dataset,
            manifest_path=args.manifest,
            config_path=args.config,
            output_dir=args.output,
            mapping_dir=args.mapping_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "distribution-calibrate":
        result = run_distribution_calibration(
            real_observations=args.real_observations,
            executions_root=args.executions_root,
            selection_manifest=args.selection_manifest,
            campaign_state=args.campaign_state,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "mmd-abc-plan":
        output = write_mmd_abc_plan(args.config, args.output)
        print(json.dumps({"plan": str(output)}, sort_keys=True))
        return
    if args.command == "mmd-abc-infer":
        result = run_mmd_abc(
            real_observations=args.real_observations,
            executions_root=args.executions_root,
            proposal_plan=args.proposal_plan,
            campaign_state=args.campaign_state,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "mmd-abc-validation-plan":
        output = write_posterior_predictive_plan(
            calibration_dir=args.calibration_dir,
            config_path=args.config,
            output_path=args.output,
        )
        print(json.dumps({"plan": str(output)}, sort_keys=True))
        return
    if args.command == "family-compare":
        result = run_family_comparison(
            primary_dir=args.primary_dir,
            candidate_dir=args.candidate_dir,
            primary_label=args.primary_label,
            candidate_label=args.candidate_label,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "prepare-upv":
        result = prepare_upv_protocol(
            archive_path=args.archive,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result["summary"], sort_keys=True))
        return
    if args.command == "analyze-upv-support":
        result = analyze_upv_support(
            route_observations=args.route_observations,
            locked_split=args.locked_split,
            upv_archive=args.upv_archive,
            phase1_config=args.phase1_config,
            selection_manifest=args.selection_manifest,
            campaign_state=args.campaign_state,
            executions_root=args.executions_root,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "audit-upv-measurement":
        result = run_measurement_equivalence_audit(
            upv_archive=args.upv_archive,
            phase2_manifest=args.phase2_manifest,
            phase2_gate=args.phase2_gate,
            oai_source=args.oai_source,
            profile_source=args.profile_source,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "plan-upv-support-v2":
        output = write_upv_support_v2_plan(
            phase3a_decision=args.phase3a_decision,
            phase3a_gate=args.phase3a_gate,
            config_path=args.config,
            output_path=args.output,
        )
        print(json.dumps({"plan": str(output), "execution_authorized": False}, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3b":
        result = analyze_phase3b_support(
            route_observations=args.route_observations,
            locked_split=args.locked_split,
            upv_archive=args.upv_archive,
            phase1_config=args.phase1_config,
            selection_manifest=args.selection_manifest,
            campaign_state=args.campaign_state,
            executions_root=args.executions_root,
            phase3a_decision=args.phase3a_decision,
            phase3a_gate=args.phase3a_gate,
            public_evidence=args.public_evidence,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "plan-upv-phase3c":
        result = build_phase3c_plan(
            phase3b_decision=args.phase3b_decision,
            phase3b_gate=args.phase3b_gate,
            oai_source=args.oai_source,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "evaluate-upv-phase3c-replay":
        output = write_deterministic_replay_evaluation(
            telemetry_path=args.telemetry,
            plan_dir=args.plan_dir,
            output_path=args.output,
        )
        print(json.dumps({"evaluation": str(output)}, sort_keys=True))
        return
    if args.command == "validate-upv-phase3c2-trace":
        result = run_phase3c2_trace_validation(
            config_path=args.config,
            phase3c1_result=args.phase3c1_result,
            sample_rate_evidence=args.sample_rate_evidence,
            oai_source=args.oai_source,
            output_dir=args.output,
            manifest_dir=args.manifest_output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "evaluate-upv-phase3c13-static-tdlb":
        output = write_static_tdlb_pilot_evaluation(
            telemetry_path=args.telemetry,
            execution_state_path=args.execution_state,
            config_path=args.config,
            output_path=args.output,
            identity_amendment_path=args.identity_amendment,
        )
        print(json.dumps({"evaluation": str(output)}, sort_keys=True))
        return
    if args.command == "evaluate-upv-phase3c14-awgn-control":
        output = write_awgn_execution_control_evaluation(
            telemetry_path=args.telemetry,
            execution_state_path=args.execution_state,
            config_path=args.config,
            tdlb_evaluation_path=args.tdlb_evaluation,
            tdlb_result_path=args.tdlb_result,
            output_path=args.output,
            identity_amendment_path=args.identity_amendment,
        )
        print(json.dumps({"evaluation": str(output)}, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3c15-support":
        result = write_phase3c15_support_analysis(
            route_observations=args.route_observations,
            locked_spatial_split=args.locked_spatial_split,
            phase3c14_telemetry=args.phase3c14_telemetry,
            phase3c14_evaluation=args.phase3c14_evaluation,
            phase3c14_result=args.phase3c14_result,
            phase3b_decision=args.phase3b_decision,
            phase3b_distribution_diagnostics=args.phase3b_distribution_diagnostics,
            phase3b_locked_validation_support=args.phase3b_locked_validation_support,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "freeze-upv-phase3d-radio-process":
        result = write_phase3d_protocol_freeze(
            archive_path=args.archive,
            phase3c15_result_path=args.phase3c15_result,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3d-radio-process":
        result = analyze_phase3d_radio_process(
            archive_path=args.archive,
            phase3c15_result_path=args.phase3c15_result,
            protocol_dir=args.protocol_dir,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "evaluate-corrected-rfsim-noise-response":
        result = write_corrected_noise_response_evaluation(
            raw_archive_path=args.raw_archive,
            telemetry_path=args.telemetry,
            execution_state_path=args.execution_state,
            protocol_path=args.protocol,
            hardware_freeze_path=args.hardware_freeze,
            analysis_spec_path=args.analysis_spec,
            development_route_means_path=args.development_route_means,
            phase3d_decision_path=args.phase3d_decision,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "freeze-upv-phase3e-radio-process":
        result = write_phase3e_protocol_freeze(
            config_path=args.config,
            phase3d_config_path=args.phase3d_config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3e-radio-process":
        result = analyze_phase3e_radio_process(
            archive_path=args.archive,
            phase3d_config_path=args.phase3d_config,
            phase3d_decision_path=args.phase3d_decision,
            corrected_noise_result_path=args.corrected_noise_result,
            protocol_dir=args.protocol_dir,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3f-exchangeability":
        result = analyze_phase3f_exchangeability(
            archive_path=args.archive,
            phase3d_config_path=args.phase3d_config,
            phase3e_result_path=args.phase3e_result,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "prepare-upv-phase3g-direct-trace":
        result = prepare_phase3g_direct_trace(
            archive_path=args.archive,
            phase3d_config_path=args.phase3d_config,
            phase3f_result_path=args.phase3f_result,
            scalar_control_result_path=args.scalar_control_result,
            corrected_noise_result_path=args.corrected_noise_result,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3g-bounded-response":
        result = analyze_phase3g_bounded_response(
            campaign_dir=args.campaign_dir,
            archive_path=args.archive,
            direct_config_path=args.direct_config,
            execution_config_path=args.execution_config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "diagnose-upv-phase3g-boundary":
        result = diagnose_phase3g_boundary(
            response_dir=args.response_dir,
            direct_trace_path=args.direct_trace,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "freeze-upv-phase3h-dynamic-staircase":
        result = freeze_phase3h_dynamic_staircase(
            config_path=args.config,
            diagnosis_path=args.diagnosis,
            execution_medians_path=args.execution_medians,
            direct_trace_path=args.direct_trace,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3h-dynamic-staircase":
        result = analyze_phase3h_dynamic_staircase(
            campaign_dir=args.campaign_dir,
            protocol_dir=args.protocol_dir,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "freeze-upv-phase3i-short-trace":
        result = freeze_phase3i_short_trace(
            config_path=args.config,
            phase3h_decision_path=args.phase3h_decision,
            phase3g_execution_medians_path=args.phase3g_execution_medians,
            phase3h_state_validation_path=args.phase3h_state_validation,
            direct_trace_path=args.direct_trace,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "recover-upv-phase3i-short-trace":
        result = recover_phase3i_short_trace(
            campaign_dir=args.campaign_dir,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "analyze-upv-phase3i-short-trace":
        result = analyze_phase3i_short_trace(
            campaign_dir=args.campaign_dir,
            protocol_dir=args.protocol_dir,
            config_path=args.config,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "freeze-upv-phase3j-full-trace":
        result = freeze_phase3j_full_trace(
            config_path=args.config,
            phase3i_decision_path=args.phase3i_decision,
            phase3g_execution_medians_path=args.phase3g_execution_medians,
            phase3h_state_validation_path=args.phase3h_state_validation,
            direct_trace_path=args.direct_trace,
            pyproject_path=args.pyproject,
            uv_lock_path=args.uv_lock,
            output_dir=args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    output = build_report(args.manifest, args.output)
    print(json.dumps({"report": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
