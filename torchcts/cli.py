# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import sys
import os
import shutil

import json
import subprocess

DEFAULT_TEST_SUITES = (
    "opinfo",
    "operators",
    "training",
    "compiler",
    "device_api",
    "autograd",
    "memory",
    "dtypes",
    "strides",
    "workloads",
    "rng",
    "serialization",
    "errors",
    "stress",
    "multi_device",
    "generated",
)


def _default_test_paths(pkg_dir=None):
    pkg_dir = pkg_dir or os.path.dirname(__file__)
    return [
        os.path.join(pkg_dir, suite)
        for suite in DEFAULT_TEST_SUITES
        if os.path.isdir(os.path.join(pkg_dir, suite))
    ]


def _project_venv_python(cwd):
    if os.name == "nt":
        return os.path.join(cwd, ".venv", "Scripts", "python.exe")
    return os.path.join(cwd, ".venv", "bin", "python")

def _maybe_reexec_project_venv():
    if os.environ.get("TORCHCTS_USE_PROJECT_VENV") != "1":
        return

    target_exe = os.path.abspath(_project_venv_python(os.getcwd()))
    if os.environ.get("_TORCHCTS_VENV_ACTIVE") == target_exe:
        return

    if not os.path.exists(target_exe):
        print(
            "Error: TORCHCTS_USE_PROJECT_VENV=1 was set, but no project .venv "
            f"python was found at {target_exe}",
            file=sys.stderr,
        )
        sys.exit(1)

    probe = subprocess.run(
        [target_exe, "-c", "import torch; import torchcts"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        print(
            "Error: project .venv exists but cannot import both torch and torchcts. "
            "Install TorchCTS into that environment or run the CLI with the intended "
            f"Python interpreter. Probe output: {detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Re-executing under {target_exe} because TORCHCTS_USE_PROJECT_VENV=1", flush=True)
    os.environ["_TORCHCTS_VENV_ACTIVE"] = target_exe
    os.environ["VIRTUAL_ENV"] = os.path.abspath(os.path.join(os.getcwd(), ".venv"))
    bindir = os.path.dirname(target_exe)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.execv(target_exe, [target_exe, "-m", "torchcts"] + sys.argv[1:])

def get_template_path(name):
    # Search in package templates
    pkg_dir = os.path.dirname(__file__)
    pkg_template = os.path.join(pkg_dir, "templates", f"manifest.{name}.py")
    if os.path.exists(pkg_template):
        return pkg_template
    # Fallback to root templates
    root_template = os.path.join(os.getcwd(), "templates", f"manifest.{name}.py")
    if os.path.exists(root_template):
        return root_template
    return None
def _discover_templates():
    """Scan the templates directory for manifest.*.py files.
    
    Returns list of (name, description) tuples, sorted by name.
    Descriptions are read from '# Description: ...' comment lines in each file.
    """
    pkg_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(pkg_dir, "templates")
    templates = []
    if os.path.isdir(templates_dir):
        for f in sorted(os.listdir(templates_dir)):
            if f.startswith("manifest.") and f.endswith(".py"):
                name = f[len("manifest."):-len(".py")]
                desc = name  # fallback
                filepath = os.path.join(templates_dir, f)
                try:
                    with open(filepath, "r", encoding="utf-8") as fh:
                        for line in fh:
                            if line.startswith("# Description:"):
                                desc = line[len("# Description:"):].strip()
                                break
                except Exception:
                    pass
                templates.append((name, desc))
    return templates

def init_manifest(template_name=None, non_interactive=False):
    manifest_path = os.path.join(os.getcwd(), "manifest.py")
    if os.path.exists(manifest_path):
        if non_interactive:
            print("manifest.py already exists. Aborting in non-interactive mode.", file=sys.stderr)
            return 1
        choice = input("manifest.py already exists. Overwrite? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("Aborted.")
            return 0

    templates = _discover_templates()
    if not templates:
        print("Error: No templates found.", file=sys.stderr)
        return 1

    template_names = [t[0] for t in templates]

    if not template_name:
        if non_interactive:
            template_name = "complete" if "complete" in template_names else template_names[0]
        else:
            print("\nSelect a template:")
            for i, (name, desc) in enumerate(templates, 1):
                default_tag = " (Default)" if name == "complete" else ""
                print(f"  [{i}] {name:<14s}— {desc}{default_tag}")
            
            try:
                val = input(f"Template [1-{len(templates)}, default 1]: ").strip()
                if val == "":
                    template_name = "complete" if "complete" in template_names else template_names[0]
                elif val.isdigit() and 1 <= int(val) <= len(templates):
                    template_name = template_names[int(val) - 1]
                else:
                    print("Invalid choice. Aborted.", file=sys.stderr)
                    return 1
            except (KeyboardInterrupt, EOFError):
                print("\nAborted.")
                return 1

    src_path = get_template_path(template_name)
    if not src_path:
        print(f"Error: Template '{template_name}' not found.", file=sys.stderr)
        return 1

    try:
        shutil.copy(src_path, manifest_path)
        print(f"Created: ./manifest.py (from manifest.{template_name}.py)")
    except Exception as e:
        print(f"Failed to copy template: {e}", file=sys.stderr)
        return 1

    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    print("Created: ./results/")
    print("\nEdit manifest.py to customize, then run:\n  torchcts run")
    return 0

def check_manifest(manifest_path=None):
    if not manifest_path:
        manifest_path = os.path.join(os.getcwd(), "manifest.py")
    if not os.path.exists(manifest_path):
        print(f"Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1
    
    # Load manifest
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("manifest", manifest_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        manifest = getattr(mod, "manifest", None)
    except Exception as e:
        print(f"Failed to parse manifest: {e}", file=sys.stderr)
        return 1

    if not isinstance(manifest, dict):
        print("Error: 'manifest' must be a dictionary in manifest.py.", file=sys.stderr)
        return 1

    from torchcts.core.manifest_schema import validate_manifest
    result = validate_manifest(manifest, base_dir=os.path.dirname(os.path.abspath(manifest_path)))

    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if result.errors:
        for error in result.errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Manifest is valid!")
    return 0

def run_coverage_command(command, strict_unknowns=False):
    from torchcts.core.coverage import (
        run_audit_command,
        run_check_command,
        run_inventory_command,
        run_materialize_command,
        run_report_command,
    )

    if command == "inventory":
        return run_inventory_command()
    if command == "audit":
        return run_audit_command()
    if command == "report":
        return run_report_command()
    if command == "materialize":
        return run_materialize_command()
    if command == "check":
        return run_check_command(strict_unknowns=strict_unknowns)
    print("Error: coverage subcommand is required.", file=sys.stderr)
    return 1


def run_triage_command(args):
    if args.triage_command != "mps":
        print("Error: triage subcommand is required.", file=sys.stderr)
        return 1
    from torchcts.core.triage import run_mps_triage

    payload = run_mps_triage(
        from_file=args.from_file,
        include_crashers=args.include_crashers,
        nodes_file=args.nodes_file,
        triage_dir=args.output_dir,
        timeout=args.timeout,
        level=args.level,
        run_nodes=not args.no_run,
        repros_only=args.repros_only,
    )
    summary_path = os.path.join(args.output_dir, "summary.md")
    classifications = payload.get("classifications", {})
    print(f"Wrote MPS triage artifacts to {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Queued nodes: {len(payload.get('queue', []))}")
    print(f"Classified failures: {len(classifications)}")
    return 0


def _print_banner():
    try:
        from torchcts import __version__
    except ImportError:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("torchcts")
    print(f"\n  TorchCTS v{__version__}")
    print("  PyTorch backend validation harness with explicit capability reporting\n")

def main():
    # Configure stdout/stderr encoding/errors to handle unicode properly
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    _maybe_reexec_project_venv()

    _print_banner()

    # Bypassing argparse entirely for the 'run' subcommand to support forwarding any unknown/arbitrary arguments to pytest.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from torchcts.core.device import _check_hardware_alignment
        if not _check_hardware_alignment():
            sys.exit(1)
        # Handle implicit init if manifest.py doesn't exist
        manifest_exists = os.path.exists(os.path.join(os.getcwd(), "manifest.py"))
        pyproject_exists = os.path.exists(os.path.join(os.getcwd(), "pyproject.toml"))
        has_toml_config = False
        if pyproject_exists:
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    tomllib = None
            if tomllib:
                try:
                    with open("pyproject.toml", "rb") as f:
                        data = tomllib.load(f)
                        if "tool" in data and ("torchcts" in data["tool"] or "backend-validator" in data["tool"]):
                            has_toml_config = True
                except Exception:
                    pass

        if not manifest_exists and not has_toml_config:
            print("No manifest.py found in the current directory.")
            non_interactive = "--non-interactive" in sys.argv or "TORCHCTS_NON_INTERACTIVE" in os.environ or "BACKEND_VALIDATOR_NON_INTERACTIVE" in os.environ
            if non_interactive:
                print("Error: manifest.py not found and running in non-interactive mode. Run 'torchcts init' first.", file=sys.stderr)
                sys.exit(1)
            choice = input("Would you like to initialize one? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                ret = init_manifest()
                if ret != 0:
                    sys.exit(ret)
            else:
                print("Aborted.")
                sys.exit(1)

        pytest_args = sys.argv[2:]
        
        # Load manifest to check show_traceback
        show_traceback = False
        manifest = {}
        manifest_path = os.path.join(os.getcwd(), "manifest.py")
        if os.path.exists(manifest_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("manifest", manifest_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                manifest = getattr(mod, "manifest", {})
                show_traceback = manifest.get("show_traceback", False)
            except Exception:
                pass

        # Check if user explicitly requested traceback via CLI flag
        cli_show_traceback = False
        if "--show-traceback" in pytest_args:
            cli_show_traceback = True
            pytest_args = [arg for arg in pytest_args if arg != "--show-traceback"]
            
        # Default to --tb=no if traceback is not explicitly requested
        if not (show_traceback or cli_show_traceback):
            has_tb_option = False
            for arg in pytest_args:
                if arg.startswith("--tb") or arg.startswith("-q"):
                    has_tb_option = True
                    break
            if not has_tb_option:
                pytest_args.append("--tb=no")

        # --- Parallel execution warning ---
        # Detect -n (pytest-xdist) and warn about GPU contention
        detected_n = 0
        for idx, a in enumerate(pytest_args):
            if a == "-n" and idx + 1 < len(pytest_args):
                try:
                    detected_n = int(pytest_args[idx + 1])
                except ValueError:
                    pass
                break
        if detected_n > 0:
            print(f"\n  ⚡ Parallel mode: {detected_n} workers")
            print("  ⚠  Some GPU drivers hang under multi-process contention.")
            print("     MPS (Apple Silicon) is known to deadlock on heavy linalg ops.")
            print("     If the run hangs, kill it — partial results are saved to disk.")
            print("     Re-run without -n to finish remaining tests.\n")

        # --- Backend selection (before pytest captures stdin) ---
        # If user didn't pass --device, detect backends now and prompt if
        # multiple are found. Pass the result to pytest via --device.
        has_device_arg = any(
            arg == "--device" or arg.startswith("--device=")
            for arg in pytest_args
        )
        if not has_device_arg and "--collect-only" not in pytest_args:
            device_name = manifest.get("device_name", "auto") if manifest_exists else "auto"
            if device_name == "auto":
                from torchcts.core.device import detect_backends
                backend_import = manifest.get("backend_import") if manifest_exists else None
                if backend_import:
                    try:
                        import importlib as _imp
                        _imp.import_module(backend_import)
                    except Exception as e:
                        print(f"Warning: Failed to import backend_import '{backend_import}': {e}", file=sys.stderr)

                detected = detect_backends()
                if len(detected) == 1:
                    device_name = detected[0][0]
                    print(f"Auto-detected backend: {device_name} ({detected[0][1]})")
                elif len(detected) > 1:
                    print("\nAvailable backends detected:")
                    for idx, entry in enumerate(detected, 1):
                        print(f"  [{idx}] {entry[0]} ({entry[1]})")
                    non_interactive = "--non-interactive" in sys.argv or "TORCHCTS_NON_INTERACTIVE" in os.environ or "BACKEND_VALIDATOR_NON_INTERACTIVE" in os.environ
                    if non_interactive:
                        backend_list = ", ".join(f"{e[0]} ({e[1]})" for e in detected)
                        print(f"Error: Ambiguous device selection in non-interactive mode. Detected: {backend_list}", file=sys.stderr)
                        sys.exit(1)
                    try:
                        choice = input(f"Select a backend [1-{len(detected)}]: ").strip()
                        if choice.isdigit() and 1 <= int(choice) <= len(detected):
                            device_name = detected[int(choice) - 1][0]
                        else:
                            print("Invalid selection.", file=sys.stderr)
                            sys.exit(1)
                    except (KeyboardInterrupt, EOFError):
                        print("\nAborted.", file=sys.stderr)
                        sys.exit(1)
                else:
                    print("Error: No device backend detected.", file=sys.stderr)
                    sys.exit(1)
            pytest_args.extend(["--device", device_name])

        has_path = False
        for arg in pytest_args:
            if not arg.startswith("-"):
                clean_arg = arg.split("::")[0]
                if os.path.exists(clean_arg) or os.path.exists(os.path.abspath(clean_arg)):
                    has_path = True
                    break
        if not has_path:
            pkg_dir = os.path.dirname(__file__)
            pytest_args.extend(_default_test_paths(pkg_dir))

        import pytest
        sys.exit(pytest.main(pytest_args))

    parser = argparse.ArgumentParser(
        description="TorchCTS — validate PyTorch backend behavior with explicit capability reporting."
    )
    
    # We want a global parser that allows running the subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # Init subcommand
    init_parser = subparsers.add_parser("init", help="Initialize manifest.py from template")

    template_choices = [name for name, _ in _discover_templates()]
    init_parser.add_argument("--template", choices=template_choices, help="Template type")
    init_parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    
    # Run subcommand placeholder for help/documentation
    subparsers.add_parser("run", help="Run the test suite")

    # Show-skips subcommand
    show_parser = subparsers.add_parser("show-skips", help="Show which tests will be skipped and why")
    show_parser.add_argument("--device", help="Target device name (e.g., mps, cuda)")
    show_parser.add_argument("--dtype", action="append", help="Target dtypes to show skips for")
    show_parser.add_argument("--level", type=int, help="Semantic level to show skips for (1-8)")
    show_parser.add_argument("--level-exact", type=int, help="Show only semantic test cases with semantic_level == LEVEL (1-8)")
    show_parser.add_argument("--level-range", help="Show only semantic test cases in inclusive MIN:MAX level range")
    
    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Regenerate reports from JSON results")
    report_parser.add_argument("--from-file", dest="from_file", help="Path to the results JSON file")

    # Sync-opinfo subcommand
    sync_parser = subparsers.add_parser("sync-opinfo", help="Force-rebuild the OpInfo registry cache")
    sync_parser.add_argument("--discover-ieee754-undefined", action="store_true",
                             help="Discover ops with undefined CPU NaN/Inf behavior")

    # Check-manifest subcommand
    check_parser = subparsers.add_parser("check-manifest", help="Validate manifest.py syntax and schema")
    check_parser.add_argument("--manifest", help="Path to manifest.py", default=None)

    # Coverage subcommands
    coverage_parser = subparsers.add_parser("coverage", help="Inventory and audit backend coverage")
    coverage_subparsers = coverage_parser.add_subparsers(dest="coverage_command", help="Coverage command")
    coverage_subparsers.add_parser("inventory", help="Build dispatcher inventory using default paths")
    coverage_subparsers.add_parser("audit", help="Build coverage audit using default paths")
    coverage_subparsers.add_parser("report", help="Render coverage report from default audit")
    coverage_subparsers.add_parser("materialize", help="Write deterministic generated coverage cases using default paths")
    coverage_check = coverage_subparsers.add_parser("check", help="Validate coverage audit consistency")
    coverage_check.add_argument("--strict-unknowns", action="store_true", help="Return nonzero if unknown surfaces remain")
    coverage_check.add_argument(
        "--fail-on-unknown",
        action="store_true",
        dest="strict_unknowns",
        help="Alias for --strict-unknowns; return nonzero if unknown surfaces remain",
    )

    # Triage subcommands
    triage_parser = subparsers.add_parser("triage", help="Crash-safe backend failure adjudication")
    triage_subparsers = triage_parser.add_subparsers(dest="triage_command", help="Triage command")
    mps_triage = triage_subparsers.add_parser("mps", help="Adjudicate MPS segfaults and failures")
    mps_triage.add_argument("--from-file", dest="from_file", help="Seed result JSON; defaults to latest MPS result")
    mps_triage.add_argument("--nodes-file", dest="nodes_file", help="Optional newline-delimited pytest node list")
    mps_triage.add_argument("--include-crashers", action="store_true", help="Include known/runlog crash candidates and run repro scripts")
    mps_triage.add_argument("--output-dir", default="results/mps_triage", help="Directory for triage artifacts")
    mps_triage.add_argument("--timeout", type=float, default=120.0, help="Seconds allowed for each subprocess run")
    mps_triage.add_argument("--level", type=int, default=8, help="Semantic level for subprocess pytest runs")
    mps_triage.add_argument("--no-run", action="store_true", help="Classify existing artifacts without executing queued pytest nodes")
    mps_triage.add_argument("--repros-only", action="store_true", help="Run standalone MPS repro scripts without executing queued pytest nodes")

    args, unknown = parser.parse_known_args()

    if not args.command:
        # Implicit init or print help
        manifest_exists = os.path.exists(os.path.join(os.getcwd(), "manifest.py"))
        if not manifest_exists:
            print("No manifest.py found in the current directory.")
            non_interactive = "--non-interactive" in sys.argv or "TORCHCTS_NON_INTERACTIVE" in os.environ or "BACKEND_VALIDATOR_NON_INTERACTIVE" in os.environ
            if non_interactive:
                print("Error: manifest.py not found and running in non-interactive mode. Run 'torchcts init' first.", file=sys.stderr)
                sys.exit(1)
            choice = input("Would you like to initialize one? [Y/n]: ").strip().lower()
            if choice in ("", "y", "yes"):
                sys.exit(init_manifest())
            else:
                parser.print_help()
                sys.exit(0)
        else:
            parser.print_help()
            sys.exit(0)

    if args.command == "init":
        sys.exit(init_manifest(args.template, args.non_interactive))

    elif args.command == "check-manifest":
        sys.exit(check_manifest(args.manifest))

    elif args.command == "coverage":
        sys.exit(run_coverage_command(args.coverage_command, getattr(args, "strict_unknowns", False)))

    elif args.command == "triage":
        sys.exit(run_triage_command(args))

    elif args.command == "sync-opinfo":
        try:
            import torch
            import torch.testing._internal.common_methods_invocations as cmi
            print(f"OpInfo database loaded: {len(cmi.op_db)} ops from PyTorch {torch.__version__}")
            if getattr(args, 'discover_ieee754_undefined', False):
                from torchcts.core.opinfo_adapter import discover_ieee754_undefined_ops, save_ieee754_undefined
                print("Discovering ops with undefined CPU NaN/Inf behavior...")
                undefined = discover_ieee754_undefined_ops()
                save_ieee754_undefined(undefined)
                print(f"Found {len(undefined)} ops with undefined NaN/Inf behavior")
                if undefined:
                    for name in sorted(undefined):
                        print(f"  - {name}")
            sys.exit(0)
        except Exception as e:
            print(f"Error loading OpInfo database: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "report":
        try:
            from torchcts.core.report import generate_report_cli
            sys.exit(generate_report_cli(args.from_file))
        except Exception as e:
            print(f"Error generating report: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "show-skips":
        # show-skips runs pytest with --show-skips --collect-only
        pytest_args = ["--collect-only", "--show-skips"]
        if args.device:
            pytest_args.extend(["--device", args.device])
        if args.dtype:
            for dt in args.dtype:
                pytest_args.extend(["--dtype", dt])
        if args.level is not None:
            pytest_args.extend(["--level", str(args.level)])
        if args.level_exact is not None:
            pytest_args.extend(["--level-exact", str(args.level_exact)])
        if args.level_range is not None:
            pytest_args.extend(["--level-range", args.level_range])
        
        import pytest
        pkg_dir = os.path.dirname(__file__)
        pytest_args.extend(_default_test_paths(pkg_dir))
        exit_code = pytest.main(pytest_args)
        # pytest returns 5 when the plugin intentionally clears all items after
        # emitting the skip audit. Treat that as success for this dry-run mode.
        sys.exit(0 if exit_code == 5 else exit_code)

if __name__ == "__main__":
    main()
