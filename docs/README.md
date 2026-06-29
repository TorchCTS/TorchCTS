# TorchCTS Documentation

Public TorchCTS documentation records stable project policy and accepted
contracts. Live coverage counts and run results are generated artifacts, not
hand-maintained documentation.

## Topics

- [Harness Runtime Policy](harness.md): manifest claim semantics, structured
  accounting, diagnostic probe evidence, crash isolation, and OpInfo oracle
  evidence.
- [Coverage](coverage/README.md): dispatcher-surface coverage policy, oracle
  authoring rules, backend-pack rules, exclusion policy, and accepted contract
  evidence.
- [Release Checklist](release.md): release validation commands, PyPI package
  README validation, backend hardware checks, and repository hygiene.

## Generated Artifacts

Use `results/coverage/` for live coverage audit output and `results/` for test
run output. Generated artifacts can change with the installed PyTorch build,
backend, manifest, and selected semantic level.

Generated artifacts are evidence. Tracked markdown files describe stable
TorchCTS policy and public release procedure.
