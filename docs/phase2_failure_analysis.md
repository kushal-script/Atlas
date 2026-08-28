# Drift Sense, Phase 2 failure analysis

Team Atlas. Every number is traceable to a run under `experiments/` in the repository; nothing is estimated.

## Where the method fails, and why

**The dominant failure is the impostor that rescales onto the lattice.** An absent pair's reference comes from a different die region of the same architecture. Because the zoom is unknown over 8 to 12, the pose search is free to rescale the impostor until its lattice period locks onto the search image's lattice, and the shared periodic content then correlates respectably. Measured over a 216 pair generated suite, absent pairs reach a median peak of 0.56, higher than genuinely present pairs at severity 2 degradation (0.52). A threshold on the correlation peak alone therefore cannot implement the found flag: its best reject class F1 on that suite is 0.46, with the degenerate never reject rule close behind, which is exactly the failure the addendum warns scores zero. Presence is decided instead by combining the peak with the evidence that does not rescale: how degenerate the correlation surface is, whether the wide pose search beat the nominal pose, and whether the deviation field, what remains after the shared periodic content is projected out, actually matched anywhere. FILL_REJECTION

**A shortcut our own generator almost taught us.** The first fitted presence model put its largest weight on the measured image noise, because in the first training suite every absent pair happened to be a clean capture, so noisy meant present. That rule would invert on any blind set whose absent pairs are degraded. The suite generator now degrades half of its absent pairs across the same severity ladder, which removed the shortcut and forced the decision onto structural evidence. We record this because it is the Phase 2 lesson in miniature: a matcher tuned to its own generator learns the generator, not the physics.

**Peer relative standout is not presence evidence.** The Phase 1 residual stage decides ties with a z score, whether one candidate stands out among its peers. An impostor site can stand out among other impostor sites by chance, and measured medians confirmed z separates nothing (3.3 present against 3.4 absent). The presence feature is instead the winning site's deviation field margin in robust units, which asks whether the site matched at all rather than whether it beat its neighbours.

**Severe degradation remains the accuracy boundary.** Localization credit on the degraded set is FILL_CREDIT_B against FILL_CREDIT_A nominal; the loss concentrates at severities 3 and 4, where dose falls to 15 to 28 percent of nominal and the polygon widths in the search capture differ from the reference by up to 20 percent. The failing cases are lattice mislocks at an essentially correct pose, the same information limited regime the Phase 1 oracle experiment characterised: supplying the true pose does not repair them, so the honest response is the found flag and a low score rather than a confident wrong centre.

**Rotation is reported to a convention, not an assumption.** The pose grid recovers rotation to a median of 0.16 degrees, but the grid angle and the required counter clockwise convention differ by exactly a sign. The sign is a single documented constant, settled against the organiser sample pairs rather than assumed from our own generator.

## The posture on rejection

False positives and false negatives are not symmetric on a tool: a false grab silently corrupts a measurement, a false reject costs one cheap rescan. Where thresholds tie on F1, this submission takes the more precise one, and the score column expresses confidence in the decision actually made, so a rejection on overwhelming evidence ranks as high as a confident detection.
