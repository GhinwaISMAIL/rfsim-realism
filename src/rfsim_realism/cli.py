from __future__ import annotations

import argparse
import json

from .distribution import run_distribution_analysis
from .distribution_calibration import run_distribution_calibration
from .family_compare import run_family_comparison
from .fetch import fetch_archive
from .mapping import run_static_mapping
from .mmd_abc import run_mmd_abc, write_mmd_abc_plan, write_posterior_predictive_plan
from .report import build_report
from .static_grid import plan_document as static_grid_plan
from .static_grid import run_campaign as run_static_grid
from .sweep import load_config, plan_document, run_campaign, write_json
from .ucc_static import build_manifest, write_manifest
from .upv_measurement_audit import run_measurement_equivalence_audit
from .upv_protocol import prepare_upv_protocol
from .upv_support import analyze_upv_support


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
    output = build_report(args.manifest, args.output)
    print(json.dumps({"report": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
