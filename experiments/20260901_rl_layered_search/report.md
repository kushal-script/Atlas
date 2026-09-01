# A layered reinforcement learning pose search, surveyed and declined before construction

## The proposal

Replace or augment the pose grid and pyramid refinement with a layered
sequential agent: state the current pose estimate and local evidence, actions
discrete steps over x, y, theta and scale plus step halving, policy learned
from generated pairs, in the style of the artificial agent registration
literature.

## What the verified literature says

Four papers anchor the field: Liao et al., AAAI 2017 (rigid 3D registration
as a Markov decision process, supervised greedy target Q, 12 discrete
actions); Ma et al., MICCAI 2017 (multimodal registration agent, 1.42 s per
case reported on server class hardware); Krebs et al., MICCAI 2017 (agent
based deformable registration trained from synthetic exact pose pairs); Ghesu
et al., TPAMI 2019 (multi scale agent search, two to three orders of magnitude
faster than exhaustive search in high dimensional parameter spaces). Every
measured win occurs under conditions that do not hold here, and each failure
of fit is checkable in the papers themselves.

First, the speed argument inverts at four degrees of freedom. The agents beat
exhaustive search where one pose evaluation is expensive and the grid is
infeasible (Liao motivates with a 60^12 cell space). Our FFT correlation
evaluates every translation of a rotation and scale cell at once in tens of
milliseconds on CPU, so the whole grid fits inside the five second budget;
an agent instead pays one network forward pass per visited pose, which on the
four core reference machine prices a basin attempt at parity with the entire
grid, per basin, before robustness is addressed.

Second, the robustness numbers (92 to 100 percent agent success against 12 to
24 for optimizers) are measured against local optimizers of cross modality
metrics that are unreliable far from alignment, never against a dense global
search of an appropriate metric. A dense search maximises its metric globally
by construction; a sequential policy over the same metric can at best match
it.

Third, no surveyed agent handles a target that is absent, and a quarter of
the Phase 2 score is the absent decision. Our presence model consumes dense
correlation surface diagnostics (prominence, candidate counts, deviation
field, quadrant consistency, and now the period aware ambiguity block) that an
agent trajectory never produces.

Fourth, the hard failure mode here is severity 4 shot noise, where the
formation matched correlation is close to the matched filter and the oracle
probe (experiments/20260830_threshold_and_oracle) shows the evidence itself,
not the search, is what fails: the true site loses to an impostor on 66
percent of severity 4 pairs even with the true pose supplied. The agent papers
demonstrate robustness to structured appearance discrepancy, not to photon
starvation, and no published reinforcement learning registration for SEM or
wafer imagery exists; industrial SEM alignment is classical template matching
with deterministic tie breaks (Hitachi, US 7,925,095 B2).

## The one transferable idea, already deployed

What survives contact with this problem is learned valuation of pose
candidates, and the pipeline already carries it in the cheapest correct form:
a fitted statistic battery re ranks the top correlation peaks (the absolute
residual override, plus the combiner diagnostics now feeding the presence
model). Deepening that layer is measurement backed; building the agent is not.
The revisit condition is recorded: an agent becomes the right tool only if the
pose space grows past a few thousand FFT cells or the evidence function stops
being computable by FFT, and the recipe then is Liao's supervised greedy
target Q learning trained from the organisers' closed form generator, layered
coarse to fine, with an action history to prevent oscillation as in Ma.
