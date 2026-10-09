"""CLI subcommands for library integration adapters.

Exposes the Serato and Rekordbox adapters as `cuepilot serato ...` /
`cuepilot rekordbox ...` commands. All mutations go through the
backup/verify/rollback lifecycle; `--dry-run` only describes what would change.
"""

from __future__ import annotations

import argparse
import json
import sys

from core.export import validate_cue_plan_dict
from core.integrations import RekordboxAdapter, SeratoAdapter

_ACTION = {
    "serato": SeratoAdapter,
    "rekordbox": RekordboxAdapter,
}


def _load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    errors = validate_cue_plan_dict(payload)
    if errors:
        print("invalid cueplan:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    return payload


def _adapter(target: str, args: argparse.Namespace):
    cls = _ACTION[target]
    kwargs = {"backups_dir": args.backups_dir}
    if target == "serato":
        kwargs["library_dir"] = getattr(args, "library_dir", None) or None
    else:
        kwargs["storage_dir"] = getattr(args, "storage_dir", None) or None
    return cls(**kwargs)


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_steps(result: dict) -> None:
    for step in result.get("steps", []):
        action = step.get("action", "?")
        detail = step.get("detail") or step.get("file") or ""
        print(f"  {action:<22} {detail}")
    if result.get("dryRun"):
        print(f"  dry run: {len(result.get('steps', []))} steps (no changes written)")


def _cmd_discover(target: str) -> None:
    def run(args: argparse.Namespace) -> int:
        adapter = _adapter(target, args)
        _print_json(adapter.discover())
        return 0

    return run


def _cmd_apply(target: str) -> None:
    def run(args: argparse.Namespace) -> int:
        plan = _load_plan(args.plan)
        adapter = _adapter(target, args)
        result = adapter.apply(plan, dry_run=args.dry_run)
        _print_json(result)
        _print_steps(result)
        return 0 if result.get("ok") else 1

    return run


def _cmd_backup(target: str) -> None:
    def run(args: argparse.Namespace) -> int:
        plan = _load_plan(args.plan)
        adapter = _adapter(target, args)
        _print_json(adapter.backup(plan))
        return 0

    return run


def _cmd_verify(target: str) -> None:
    def run(args: argparse.Namespace) -> int:
        adapter = _adapter(target, args)
        _print_json(adapter.verify(args.backup_id))
        return 0

    return run


def _cmd_rollback(target: str) -> None:
    def run(args: argparse.Namespace) -> int:
        adapter = _adapter(target, args)
        _print_json(adapter.rollback(args.backup_id))
        return 0

    return run


def add_integration_parser(sub: argparse._SubParsersAction) -> None:
    for target, cls in _ACTION.items():
        p = sub.add_parser(target, help=f"{cls.__name__} integration commands")
        p.add_argument("--backups-dir", default=None, help="backup storage directory")
        if target == "serato":
            p.add_argument("--library-dir", default=None, help="Serato `_Serato_` folder (default: auto-detect)")
        else:
            p.add_argument("--storage-dir", default=None, help="Rekordbox agent storage folder (default: auto-detect)")
        p_sub = p.add_subparsers(dest="integrate_cmd", required=True)

        p_discover = p_sub.add_parser("discover", help="locate the library and report status")
        p_discover.set_defaults(func=_cmd_discover(target))

        p_apply = p_sub.add_parser("apply", help="apply a cueplan through the safe lifecycle")
        p_apply.add_argument("plan", help="cueplan.json path")
        p_apply.add_argument("--dry-run", action="store_true", help="describe changes without writing")
        p_apply.set_defaults(func=_cmd_apply(target))

        p_backup = p_sub.add_parser("backup", help="backup the library + cueplan without applying")
        p_backup.add_argument("plan", help="cueplan.json path")
        p_backup.set_defaults(func=_cmd_backup(target))

        p_verify = p_sub.add_parser("verify", help="verify a backup against the current library")
        p_verify.add_argument("backup_id", help="backup id")
        p_verify.set_defaults(func=_cmd_verify(target))

        p_rollback = p_sub.add_parser("rollback", help="restore the library from a backup")
        p_rollback.add_argument("backup_id", help="backup id")
        p_rollback.set_defaults(func=_cmd_rollback(target))