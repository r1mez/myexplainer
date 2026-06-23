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
- Task 4: complete (commits fb1ba51..61ab310, review clean with findings noted)
  Important: baseline_eval_metrics.py fidelity/sparsity wrappers re-implement rather than delegate (API difference is genuine constraint, semantically equivalent)
  Minor: redundant conditional in compute_proximity_from_edge_index
  Minor: local import shadowing module-level Batch import
  Minor: unused fidelity/sparsity imports in baseline_eval_metrics.py
- Task 5: complete (commits 61ab310..10ca617, review clean with findings noted)
  Important: behavioral change - 6 datasets now generate patterns instead of empty lists (improvement, not bug)
  Minor: case_study/tsne_indistribution_vis.py still hardcoded (out of scope)
  Minor: baseline models still hardcode GNN paths (out of scope)
- Task 6: complete (commits 10ca617..dfdaecd, review clean with 1 Minor noted)
  Minor: missing trailing newline in subgraph_method.py
- Task 7: pending (Extract Baseline Runner — optional)
