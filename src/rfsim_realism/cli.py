from __future__ import annotations

import argparse
import json

from .distribution import run_distribution_analysis
from .fetch import fetch_archive
from .mapping import run_static_mapping
from .report import build_report
from .static_grid import plan_document as static_grid_plan
from .static_grid import run_campaign as run_static_grid
from .sweep import load_config, plan_document, run_campaign, write_json
from .ucc_static import build_manifest, write_manifest


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
    output = build_report(args.manifest, args.output)
    print(json.dumps({"report": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
