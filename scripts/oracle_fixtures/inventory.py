#!/usr/bin/env python3
"""Generate and verify the complete TorchCTS oracle inventory.

The output is intentionally deterministic: it contains no timestamps, absolute
paths, or host-specific values.  CI compares it byte-for-byte with the reviewed
snapshot under tests/oracles/cases.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchcts.core.non_unique_output_compare import NON_UNIQUE_OUTPUT_CONTRACTS
from scripts.oracle_fixtures.oracle_inventory import (
    BACKWARD_FUNCTION_CASE_PACKS,
    BACKWARD_REFERENCE_CASE_PACKS,
    DIRECT_ORACLE_CASE_PACKS,
    FFT_FUNCTION_CASE_PACKS,
    FORWARD_REFERENCE_CASE_PACKS,
    HIGH_PRECISION_FUNCTION_CASE_PACKS,
    NON_UNIQUE_CASE_PACKS,
    NON_UNIQUE_VALIDATION_CLASSES,
    REFERENCE_FUNCTION_CASE_PACKS,
    direct_validation_class,
)
from torchcts.core.oracles import all_oracle_specs


REFERENCE_MODULE_PATHS = {
    "torchcts.core.reference_oracles": REPO_ROOT / "torchcts/core/reference_oracles.py",
    "torchcts.core.high_precision_reference": REPO_ROOT / "torchcts/core/high_precision_reference.py",
    "torchcts.core.backward_references": REPO_ROOT / "torchcts/core/backward_references.py",
    "torchcts.core.fft_contract": REPO_ROOT / "torchcts/core/fft_contract.py",
    "torchcts.core.contract_references": REPO_ROOT / "torchcts/core/contract_references.py",
    "torchcts.core.non_unique_output_compare": REPO_ROOT / "torchcts/core/non_unique_output_compare.py",
    "torchcts.core.oracles": REPO_ROOT / "torchcts/core/oracles.py",
}

SNAPSHOT_PATH = REPO_ROOT / "tests/oracles/cases/inventory.json"
REVIEWED_LOCAL_PATH = REPO_ROOT / "tests/oracles/cases/reviewed_local_expected.json"
SEMANTIC_MARKER_KEYWORDS = ("oracle", "semantic", "broken cpu", "absent cpu")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=_relative(path))


def _public_functions(path: Path) -> list[str]:
    return sorted(
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )


def _constant_strings(node: ast.AST) -> Iterable[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _forward_reference_ids() -> list[str]:
    tree = _tree(REFERENCE_MODULE_PATHS["torchcts.core.contract_references"])
    result: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            result.add(node.args[0].value)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "_OPINFO_COMPLEX_UNARY",
            "_GENERATED_COMPLEX_UNARY",
        }:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for value in node.value.values:
            if (
                isinstance(value, ast.Tuple)
                and value.elts
                and isinstance(value.elts[0], ast.Constant)
                and isinstance(value.elts[0].value, str)
            ):
                result.add(value.elts[0].value)
    return sorted(result)


def _backward_reference_ids() -> list[str]:
    tree = _tree(REFERENCE_MODULE_PATHS["torchcts.core.backward_references"])
    result = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BackwardContract"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            result.add(node.args[0].value)
    return sorted(result)


def _fft_families() -> dict[str, list[str]]:
    tree = _tree(REFERENCE_MODULE_PATHS["torchcts.core.fft_contract"])
    wanted = {"_ONE_DIMENSIONAL", "_TWO_DIMENSIONAL", "_C2C", "_R2C", "_C2R", "_R2H"}
    result: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        result[target.id.removeprefix("_").lower()] = sorted(set(_constant_strings(node.value)))
    result["generated"] = ["_fft_c2c"]
    return dict(sorted(result.items()))


def _module_public_callables() -> dict[str, set[str]]:
    return {module: set(_public_functions(path)) for module, path in REFERENCE_MODULE_PATHS.items()}


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    result: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        for line in range(node.lineno, end + 1):
            result[line] = node.name
    return result


def _call_sites() -> list[dict]:
    public = _module_public_callables()
    rows: list[dict] = []
    for path in sorted((REPO_ROOT / "torchcts").rglob("*.py")):
        tree = _tree(path)
        enclosing = _enclosing_functions(tree)
        aliases: dict[str, tuple[str, str]] = {}
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in public:
                for imported in node.names:
                    if imported.name in public[node.module]:
                        aliases[imported.asname or imported.name] = (node.module, imported.name)
            elif isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name in public and imported.asname:
                        module_aliases[imported.asname] = imported.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            module = name = None
            if isinstance(node.func, ast.Name) and node.func.id in aliases:
                module, name = aliases[node.func.id]
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
            ):
                module = module_aliases[node.func.value.id]
                if node.func.attr in public[module]:
                    name = node.func.attr
            if module is None or name is None:
                continue
            rows.append({
                "path": _relative(path),
                "line": node.lineno,
                "function": enclosing.get(node.lineno, "<module>"),
                "module": module,
                "callable": name,
            })
    return sorted(rows, key=lambda row: (row["path"], row["line"], row["callable"]))


def _string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_cpu_contract_exempt_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cpu_contract_exempt"
    )


def _semantic_marked_tests() -> list[dict]:
    rows: list[dict] = []
    for path in sorted((REPO_ROOT / "torchcts").rglob("test_*.py")):
        tree = _tree(path)
        semantic_aliases: dict[str, str] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or not _is_cpu_contract_exempt_call(node.value):
                continue
            reason = _string_value(node.value.args[0]) if node.value.args else ""
            if reason and any(keyword in reason.lower() for keyword in SEMANTIC_MARKER_KEYWORDS):
                semantic_aliases[node.targets[0].id] = reason
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            reasons = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id in semantic_aliases:
                    reasons.append(semantic_aliases[decorator.id])
                elif _is_cpu_contract_exempt_call(decorator):
                    reason = _string_value(decorator.args[0]) if decorator.args else ""
                    if reason and any(keyword in reason.lower() for keyword in SEMANTIC_MARKER_KEYWORDS):
                        reasons.append(reason)
            if reasons:
                rows.append({
                    "path": _relative(path),
                    "line": node.lineno,
                    "function": node.name,
                    "reason": sorted(set(reasons))[0],
                })
    return sorted(rows, key=lambda row: (row["path"], row["line"]))


def _assignment_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    targets = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def _suspicious_local_expected() -> list[dict]:
    """Return a review queue for hand-authored expected/reference calculations."""

    rows: list[dict] = []
    semantic = {(row["path"], row["function"]): row for row in _semantic_marked_tests()}
    for path in sorted((REPO_ROOT / "torchcts").rglob("test_*.py")):
        tree = _tree(path)
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            if not function.name.startswith("test_"):
                continue
            assignments = []
            for node in ast.walk(function):
                names = _assignment_names(node)
                if any("expected" in name.lower() or "reference" in name.lower() for name in names):
                    assignments.append(node.lineno)
            key = (_relative(path), function.name)
            if assignments and key in semantic:
                rows.append({
                    "path": key[0],
                    "function": key[1],
                    "line": function.lineno,
                    "assignment_lines": sorted(set(assignments)),
                    "reason": "semantic CPU-contract exemption with local expected/reference calculation",
                })
    return sorted(rows, key=lambda row: (row["path"], row["line"]))


def _direct_specs() -> list[dict]:
    return [
        {
            "surface": spec.surface,
            "oracle_id": spec.oracle_id,
            "coverage_status": spec.coverage_status,
            "coverage_kind": spec.coverage_kind,
            "runner": spec.runner,
            "backend_gate": spec.backend_gate,
            "contract_status": spec.metadata()["contract_status"],
            "contract_ref": spec.contract_ref,
            "case_pack": DIRECT_ORACLE_CASE_PACKS[spec.oracle_id],
            "validation_class": direct_validation_class(spec.oracle_id, spec.coverage_status),
        }
        for spec in all_oracle_specs()
    ]


def _mapping_errors(discovered: dict) -> list[str]:
    errors = []
    expected = {
        "reference_functions": REFERENCE_FUNCTION_CASE_PACKS,
        "high_precision_functions": HIGH_PRECISION_FUNCTION_CASE_PACKS,
        "backward_functions": BACKWARD_FUNCTION_CASE_PACKS,
        "fft_functions": FFT_FUNCTION_CASE_PACKS,
        "forward_reference_ids": FORWARD_REFERENCE_CASE_PACKS,
        "backward_reference_ids": BACKWARD_REFERENCE_CASE_PACKS,
    }
    for key, mapping in expected.items():
        names = set(discovered[key])
        if names != set(mapping):
            errors.append(
                f"{key} mapping mismatch: missing={sorted(names - set(mapping))} "
                f"orphan={sorted(set(mapping) - names)}"
            )
    non_unique = set(discovered["non_unique_contracts"])
    if non_unique != set(NON_UNIQUE_CASE_PACKS) or non_unique != set(NON_UNIQUE_VALIDATION_CLASSES):
        errors.append("non-unique contract mappings do not exactly match the runtime registry")
    direct_ids = {row["oracle_id"] for row in discovered["direct_specs"]}
    if direct_ids != set(DIRECT_ORACLE_CASE_PACKS):
        errors.append(
            f"direct oracle mapping mismatch: missing={sorted(direct_ids - set(DIRECT_ORACLE_CASE_PACKS))} "
            f"orphan={sorted(set(DIRECT_ORACLE_CASE_PACKS) - direct_ids)}"
        )
    return errors


def build_inventory() -> dict:
    discovered = {
        "reference_functions": _public_functions(REFERENCE_MODULE_PATHS["torchcts.core.reference_oracles"]),
        "high_precision_functions": _public_functions(REFERENCE_MODULE_PATHS["torchcts.core.high_precision_reference"]),
        "backward_functions": _public_functions(REFERENCE_MODULE_PATHS["torchcts.core.backward_references"]),
        "fft_functions": _public_functions(REFERENCE_MODULE_PATHS["torchcts.core.fft_contract"]),
        "forward_reference_ids": _forward_reference_ids(),
        "backward_reference_ids": _backward_reference_ids(),
        "fft_families": _fft_families(),
        "non_unique_contracts": sorted(contract.family for contract in NON_UNIQUE_OUTPUT_CONTRACTS),
        "direct_specs": _direct_specs(),
        "call_sites": _call_sites(),
        "semantic_marked_tests": _semantic_marked_tests(),
        "suspicious_local_expected": _suspicious_local_expected(),
    }
    errors = _mapping_errors(discovered)
    if errors:
        raise RuntimeError("\n".join(errors))
    grouped_direct = {
        "surface_count": len(discovered["direct_specs"]),
        "oracle_id_count": len({row["oracle_id"] for row in discovered["direct_specs"]}),
        "status_group_count": len({
            (
                row["oracle_id"],
                row["coverage_status"],
                row["runner"],
                row["backend_gate"],
                row["contract_status"],
                row["contract_ref"],
            )
            for row in discovered["direct_specs"]
        }),
    }
    return {
        "schema_version": 1,
        "counts": {
            "reference_functions": len(discovered["reference_functions"]),
            "high_precision_functions": len(discovered["high_precision_functions"]),
            "backward_functions": len(discovered["backward_functions"]),
            "fft_functions": len(discovered["fft_functions"]),
            "forward_reference_ids": len(discovered["forward_reference_ids"]),
            "backward_reference_ids": len(discovered["backward_reference_ids"]),
            "non_unique_contracts": len(discovered["non_unique_contracts"]),
            "direct_oracle_ids": grouped_direct["oracle_id_count"],
            "direct_status_groups": grouped_direct["status_group_count"],
            "direct_surfaces": grouped_direct["surface_count"],
            "call_sites": len(discovered["call_sites"]),
            "semantic_marked_tests": len(discovered["semantic_marked_tests"]),
            "suspicious_local_expected": len(discovered["suspicious_local_expected"]),
        },
        "case_pack_mappings": {
            "reference_functions": dict(sorted(REFERENCE_FUNCTION_CASE_PACKS.items())),
            "high_precision_functions": dict(sorted(HIGH_PRECISION_FUNCTION_CASE_PACKS.items())),
            "backward_functions": dict(sorted(BACKWARD_FUNCTION_CASE_PACKS.items())),
            "fft_functions": dict(sorted(FFT_FUNCTION_CASE_PACKS.items())),
            "forward_reference_ids": dict(sorted(FORWARD_REFERENCE_CASE_PACKS.items())),
            "backward_reference_ids": dict(sorted(BACKWARD_REFERENCE_CASE_PACKS.items())),
            "non_unique_contracts": dict(sorted(NON_UNIQUE_CASE_PACKS.items())),
            "direct_oracle_ids": dict(sorted(DIRECT_ORACLE_CASE_PACKS.items())),
        },
        "validation_classes": {
            "non_unique_contracts": dict(sorted(NON_UNIQUE_VALIDATION_CLASSES.items())),
        },
        "inventory": discovered,
    }


def _canonical(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _check_reviewed_local(inventory: dict) -> list[str]:
    if not REVIEWED_LOCAL_PATH.exists():
        return [f"missing reviewed local-expected file: {_relative(REVIEWED_LOCAL_PATH)}"]
    reviewed = json.loads(REVIEWED_LOCAL_PATH.read_text(encoding="utf-8"))
    expected = reviewed.get("reviewed", [])
    actual = inventory["inventory"]["suspicious_local_expected"]
    if expected != actual:
        return [
            "reviewed local-expected audit is stale; run inventory.py --json and "
            "review every suspicious_local_expected entry"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the canonical inventory")
    parser.add_argument("--check", action="store_true", help="verify the checked inventory snapshot")
    args = parser.parse_args(argv)
    inventory = build_inventory()
    if args.json:
        print(_canonical(inventory), end="")
    if args.check:
        errors = _check_reviewed_local(inventory)
        if not SNAPSHOT_PATH.exists():
            errors.append(f"missing inventory snapshot: {_relative(SNAPSHOT_PATH)}")
        elif SNAPSHOT_PATH.read_text(encoding="utf-8") != _canonical(inventory):
            errors.append("oracle inventory snapshot is stale; regenerate and review it")
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        counts = inventory["counts"]
        print(
            "oracle inventory verified: "
            f"{counts['reference_functions']} core functions, "
            f"{counts['forward_reference_ids']} forward IDs, "
            f"{counts['backward_reference_ids']} backward IDs, "
            f"{counts['non_unique_contracts']} non-unique contracts, "
            f"{counts['direct_surfaces']} direct surfaces"
        )
    if not args.json and not args.check:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
