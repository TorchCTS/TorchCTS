import argparse
import sys
import os
import shutil
import importlib
import json

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

    templates = ["complete", "training", "inference", "minimal"]
    if not template_name:
        if non_interactive:
            template_name = "complete"
        else:
            print("\nSelect a template:")
            print("  [1] complete    — broadest template, intended for explicit opt-in only (Default)")
            print("  [2] training    — full training loop: autograd, optimizers, autocast, DataLoader")
            print("  [3] inference   — production inference, all dtypes, full OpInfo sweep")
            print("  [4] minimal     — device registration + ~20 core ops (float32/int64/bool only)")
            
            try:
                val = input("Template [1-4, default 1]: ").strip()
                if val == "":
                    template_name = "complete"
                elif val in ("1", "2", "3", "4"):
                    template_name = templates[int(val) - 1]
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

    # Validate keys
    required_keys = ["manifest_version", "device_name", "capabilities"]
    for k in required_keys:
        if k not in manifest:
            print(f"Error: Missing required key '{k}'", file=sys.stderr)
            return 1

    if manifest.get("manifest_version") != 1:
        print(f"Warning: Unexpected manifest_version {manifest.get('manifest_version')}. Expected 1.", file=sys.stderr)

    # Validate capabilities
    caps = manifest.get("capabilities", {})
    if not isinstance(caps, dict):
        print("Error: 'capabilities' must be a dictionary.", file=sys.stderr)
        return 1

    # Validate hardware section
    hw = manifest.get("hardware", {})
    if not isinstance(hw, dict):
        print("Error: 'hardware' must be a dictionary.", file=sys.stderr)
        return 1
    
    if "system_memory_gb" in hw:
        sys_mem = hw["system_memory_gb"]
        if sys_mem != "auto":
            if not isinstance(sys_mem, (int, float)) or sys_mem <= 0:
                print("Error: 'system_memory_gb' must be 'auto' or a positive number.", file=sys.stderr)
                return 1
                
    if "device_memory_gb" in hw:
        dev_mem = hw["device_memory_gb"]
        if dev_mem != "auto":
            if not isinstance(dev_mem, list) or not all(isinstance(x, (int, float)) and x > 0 for x in dev_mem):
                print("Error: 'device_memory_gb' must be 'auto' or a list of positive numbers.", file=sys.stderr)
                return 1

    print("Manifest is valid!")
    return 0

def _print_banner():
    from torchcts import __version__
    print(f"\n  TorchCTS v{__version__}")
    print("  PyTorch backend validation harness with explicit capability reporting\n")

def main():
    # Configure stdout/stderr encoding/errors to handle unicode properly
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # Dynamic re-exec to run under the local venv python if present in the current working directory
    local_venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python")
    if os.path.exists(local_venv_python) and os.environ.get("_TORCHCTS_VENV_ACTIVE") != "1":
        target_exe = os.path.abspath(local_venv_python)
        os.environ["_TORCHCTS_VENV_ACTIVE"] = "1"
        os.environ["PYTHONPATH"] = os.path.abspath(os.getcwd())
        os.environ["VIRTUAL_ENV"] = os.path.abspath(os.path.join(os.getcwd(), ".venv"))
        os.environ["PATH"] = os.path.abspath(os.path.join(os.getcwd(), ".venv", "bin")) + os.pathsep + os.environ.get("PATH", "")
        os.execv(target_exe, [target_exe, "-m", "torchcts"] + sys.argv[1:])

    _print_banner()

    # Bypassing argparse entirely for the 'run' subcommand to support forwarding any unknown/arbitrary arguments to pytest.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
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
                    non_interactive = "--non-interactive" in sys.argv or os.environ.get("BACKEND_VALIDATOR_NON_INTERACTIVE") == "1"
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
            pytest_args.append(pkg_dir)

        import pytest
        sys.exit(pytest.main(pytest_args))

    parser = argparse.ArgumentParser(
        description="TorchCTS — validate PyTorch backend behavior with explicit capability reporting."
    )
    
    # We want a global parser that allows running the subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")
    
    # Init subcommand
    init_parser = subparsers.add_parser("init", help="Initialize manifest.py from template")
    init_parser.add_parser_argument = init_parser.add_argument  # compatibility/convenience
    init_parser.add_argument("--template", choices=["minimal", "inference", "training", "complete"], help="Template type")
    init_parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    
    # Run subcommand placeholder for help/documentation
    subparsers.add_parser("run", help="Run the test suite")

    # Show-skips subcommand
    show_parser = subparsers.add_parser("show-skips", help="Show which tests will be skipped and why")
    show_parser.add_argument("--device", help="Target device name (e.g., mps, cuda)")
    show_parser.add_argument("--dtype", action="append", help="Target dtypes to show skips for")
    
    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Regenerate reports from JSON results")
    report_parser.add_argument("--from-file", dest="from_file", help="Path to the results JSON file")

    # Sync-opinfo subcommand
    subparsers.add_parser("sync-opinfo", help="Force-rebuild the OpInfo registry cache")

    # Check-manifest subcommand
    check_parser = subparsers.add_parser("check-manifest", help="Validate manifest.py syntax and schema")
    check_parser.add_argument("--manifest", help="Path to manifest.py", default=None)

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

    elif args.command == "sync-opinfo":
        try:
            import torch
            import torch.testing._internal.common_methods_invocations as cmi
            print(f"OpInfo database loaded: {len(cmi.op_db)} ops from PyTorch {torch.__version__}")
            from torchcts.core.opinfo_adapter import load_known_failures
            known = load_known_failures()
            fwd_count = len(known.get("forward", {}))
            bwd_count = len(known.get("backward", {}))
            print(f"Known CPU failures: {fwd_count} forward, {bwd_count} backward")
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
        
        import pytest
        pkg_dir = os.path.dirname(__file__)
        pytest_args.append(pkg_dir)
        exit_code = pytest.main(pytest_args)
        # pytest returns 5 when the plugin intentionally clears all items after
        # emitting the skip audit. Treat that as success for this dry-run mode.
        sys.exit(0 if exit_code == 5 else exit_code)

if __name__ == "__main__":
    main()
