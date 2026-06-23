# Architecture Refactor — Progress Ledger

Base commit: 411b935

- Task 1: complete (commits 411b935..1034586, review clean with 2 Minor findings noted)
  Minor: evaluationV2.py has 7 unused imports left over from deleted code
  Minor: clear.py has residual `# Active implementation starts here.` comment
- Task 2: complete (commits 1034586..6d2e863, review clean with 3 Minor findings noted)
  Minor: config/explainer_config.py comment says "CLI args override YAML" but behavior is opposite
  Minor: unused `from typing import Optional` import
  Minor: evaluationV2.py docstring still references `args` instead of `config`
- Task 3: complete (commits 6d2e863..fb1ba51, review found 2 Critical callers fixed)
  Critical: c2explainer.py and atex_cf.py had edge_mask= keyword args -> fixed to edge_weight=
- Task 4: pending (Consolidate Evaluation Metrics)
- Task 5: pending (Dataset Registry)
- Task 6: pending (Clean Up Subgraph Mining)
- Task 7: pending (Extract Baseline Runner — optional)
