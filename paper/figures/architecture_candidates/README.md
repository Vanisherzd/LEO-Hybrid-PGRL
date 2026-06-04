# Architecture candidates

These alternatives are not wired into the main paper yet. They exist only for
review and future selection.

- `candidate_a_three_layer.pdf`: simplified three-layer system view
- `candidate_b_closed_loop.pdf`: circular control/evidence loop
- `candidate_c_swimlane.pdf`: offline/online/hardware swimlane

Current recommendation: keep the current in-paper Fig. 1 for now. Among the
alternatives, **Candidate C** is the strongest fallback because its swimlane
structure separates offline, online, and hardware paths more cleanly than
Candidate A, while Candidate B is easier to scan but drops too much control-path
detail.
