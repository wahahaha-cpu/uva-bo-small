#!/usr/bin/env python3
"""Validate an isolated LIBERO-Plus installation before a long evaluation."""

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path


EXPECTED_TASKS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
    "libero_90": 90,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10", choices=sorted(EXPECTED_TASKS))
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="report missing generated assets/BDDL files without failing",
    )
    args = parser.parse_args()

    try:
        import libero
        import libero.libero as libero_core
        from libero.libero import benchmark, get_libero_path
    except Exception as exc:  # pragma: no cover - exercised by environment setup
        print(f"ERROR: cannot import LIBERO-Plus: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    module_path = getattr(libero, "__file__", None) or libero_core.__file__
    print(f"libero module: {Path(module_path).resolve()}")
    paths = {key: Path(get_libero_path(key)).expanduser() for key in (
        "benchmark_root", "bddl_files", "init_states", "datasets", "assets"
    )}
    for key, path in paths.items():
        print(f"{key}: {path} ({'present' if path.exists() else 'MISSING'})")

    failures = []
    if not paths["bddl_files"].is_dir():
        failures.append(f"missing bddl_files directory: {paths['bddl_files']}")
    if not paths["init_states"].is_dir():
        failures.append(f"missing init_states directory: {paths['init_states']}")
    assets_children = list(paths["assets"].iterdir()) if paths["assets"].is_dir() else []
    if not assets_children:
        failures.append(
            f"LIBERO-Plus assets are missing: {paths['assets']} "
            "(download assets.zip and extract it there)"
        )

    suite_cls = benchmark.get_benchmark_dict()[args.suite]
    with contextlib.redirect_stdout(io.StringIO()):
        suite = suite_cls(task_order_index=0)
    actual_tasks = suite.n_tasks
    expected_tasks = EXPECTED_TASKS[args.suite]
    print(f"suite {args.suite}: {actual_tasks} tasks (expected {expected_tasks})")
    if actual_tasks != expected_tasks:
        failures.append(f"unexpected task count for {args.suite}: {actual_tasks} != {expected_tasks}")

    classification_path = paths["benchmark_root"] / "benchmark" / "task_classification.json"
    if classification_path.is_file():
        classification = json.loads(classification_path.read_text())
        if args.suite in classification:
            counts = Counter(item["category"] for item in classification[args.suite])
            print(f"classification entries: {len(classification[args.suite])}; categories: {dict(counts)}")

    bddl_missing = []
    for task_id in range(suite.n_tasks):
        path = Path(suite.get_task_bddl_file_path(task_id))
        # Plus encodes camera and robot-state perturbations after the base BDDL
        # name. The Plus environment strips this suffix at construction time.
        if not path.is_file() and "_view_" in path.stem:
            base_stem = path.stem.split("_view_", 1)[0]
            encoded_base = path.with_name(base_stem + ".bddl")
            if encoded_base.is_file():
                continue
        if not path.is_file():
            bddl_missing.append(path.name)
    print(f"missing BDDL files for {args.suite}: {len(bddl_missing)}")
    if bddl_missing:
        failures.append(f"{len(bddl_missing)} generated BDDL files are missing")

    # Loading one initial state catches wrong LIBERO_CONFIG_PATH and missing generated states.
    try:
        initial_states = suite.get_task_init_states(0)
        print(f"sample initial states: loaded shape={getattr(initial_states, 'shape', None)}")
    except Exception as exc:
        failures.append(f"sample initial state failed: {type(exc).__name__}: {exc}")

    if failures and not args.allow_missing_data:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    for failure in failures:
        print(f"WARNING: {failure}")
    print("LIBERO-Plus preflight complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
