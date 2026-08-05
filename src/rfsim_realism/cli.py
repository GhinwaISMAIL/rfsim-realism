from __future__ import annotations

import argparse
import json

from .fetch import fetch_archive
from .report import build_report
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
    output = build_report(args.manifest, args.output)
    print(json.dumps({"report": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
