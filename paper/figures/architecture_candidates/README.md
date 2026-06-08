# Architecture candidates

These alternatives are not wired into the main paper yet. They exist only for
review and future selection.

- `candidate_a_three_layer.pdf`: simplified three-layer system view
- `candidate_b_closed_loop.pdf`: circular control/evidence loop
- `candidate_c_swimlane.pdf`: offline/online/hardware swimlane
- `candidate_d_pgrl_control_stack.pdf`: PGRL-centered control-stack view

Current recommendation: keep the current in-paper Fig. 1 for now. Among the
alternatives, **Candidate D** is the strongest future direction because it makes
PGRL central and keeps the uncertainty-to-control mapping explicit without the
extra offline-training detail. Candidate C remains the cleanest swimlane option.
