#!/usr/bin/env python3
"""Generate the checked-in path-shape corpus from family specs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchcts.path_shapes.loader import CORPUS_PATH
from torchcts.path_shapes.specs import SPEC_MODULES
from torchcts.path_shapes.specs.common import base_corpus
from torchcts.path_shapes.validation import PathShapeValidationError, validate_path_shape_corpus


def generate_corpus() -> dict:
    cases = []
    for module in SPEC_MODULES:
        cases.extend(module.cases())

    seen: dict[str, str] = {}
    duplicates = []
    for case in cases:
        case_id = case["case_id"]
        runner = case["runner"]
        if case_id in seen:
            duplicates.append(f"{case_id} ({seen[case_id]} and {runner})")
        seen[case_id] = runner
    if duplicates:
        joined = "\n".join(sorted(duplicates))
        raise SystemExit(f"duplicate generated path-shape case IDs:\n{joined}")

    return base_corpus(cases)


def _serialized(corpus: dict) -> str:
    return json.dumps(corpus, indent=2, sort_keys=False) + "\n"


def _summary_lines(summary: dict) -> list[str]:
    return [
        f"Path-shape cases: {summary['case_count']}",
        f"Default-selected cases: {summary['default_selected_case_count']}",
        f"Resource tiers: {', '.join(f'{key}={value}' for key, value in summary['by_resource_tier'].items())}",
        f"Families: {', '.join(f'{key}={value}' for key, value in summary['by_family'].items())}",
        f"Cost classes: {', '.join(f'{key}={value}' for key, value in summary['by_cost_class'].items())}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(CORPUS_PATH), help="Corpus path to write/check")
    parser.add_argument("--check", action="store_true", help="Fail if the output path is not up to date")
    parser.add_argument("--update", action="store_true", help="Write the generated corpus")
    parser.add_argument("--json", action="store_true", help="Print validation summary as JSON")
    parser.add_argument("--strict-budget", action="store_true", help="Enforce ratio budget against the collection baseline")
    parser.add_argument("--enforce-targets", action="store_true", help="Treat target gaps as failures")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    corpus = generate_corpus()
    try:
        summary = validate_path_shape_corpus(
            corpus,
            strict_budget=args.strict_budget,
            enforce_targets=args.enforce_targets,
        )
    except PathShapeValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    text = _serialized(corpus)
    if args.check:
        existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if existing != text:
            print(f"{output_path} is not up to date; run scripts/generate_path_shape_corpus.py --update", file=sys.stderr)
            return 1

    if args.update:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("\n".join(_summary_lines(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
