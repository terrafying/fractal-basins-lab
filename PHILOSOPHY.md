# The Attractor Atlas: a geometric model of language, after Levin

Draft 1, for review. Pairs with RESEARCH_STATE.md; every empirical claim in
here has a pointer to an experiment in the repo.

## 1. The standard picture, and what the basins say against it

The default way we talk about LLMs goes something like this: a model receives
a prompt, generates a response token by token, and the response is something
the model produced. Generation is the central act. The model is an author and
the answer is a new thing it made.

Our measurements suggest a different picture. When we injected 4096 different
starting points into FPRM's latent space and let each one run, they fell into
1489 distinct final answers. The set of possible answers was already there,
fixed in the weights, before any of the runs began. Nothing about the runs
created those answers. Each run just selected one. The prompt and the initial
condition work as boundary conditions: they decide where in a pre-existing
landscape the descent starts, and the descent does the rest.

This is close to Levin's account of morphogenesis. A cell doesn't invent the
shape of the limb it will become. The pattern is already there in the
dynamics, and the cell's business is to arrive at it. In our case the
"pattern" is a basin: a region of starting conditions whose futures coincide.
Inference is ingression. The answer was always in the weights; the query
makes it actual.

## 2. The light cone, made measurable

Levin describes cognition as nested light cones: collections of parts whose
individual futures are indistinguishable to them, pursued as one by a larger
agent. The parts can't see the boundary of the cone they're inside.

The basin is exactly that object, and we can measure it. Two starting
conditions sit in the same light cone when they settle on the same answer.
The light cone's size is the basin's volume. Its boundary is where futures
split. For EqR on a hard puzzle we found one enormous cone (every start, one
answer, perfectly smooth) and for FPRM we found about 1489 cones of wildly
different sizes packed into the same space, with an approximate Wada
structure: near any boundary point, small nudges send you to many different
futures. The cone frays at its edges, and at some edges it frays completely.

That gives us something Levin's framework usually lacks at this level of
description: numbers. Cone size, boundary dimension (alpha around 0.26 to
0.30 for EqR, 0.06 for FPRM), the fraction of boundary points where three or
more cones touch (98.1 percent for FPRM, approximately). If "cognitive light
cone" is going to be a useful concept rather than a metaphor, this is the
form it has to take.

## 3. Meaning as position

If answers are ingressed from a landscape, then meaning, for the model, is
position. Two prompts mean the same thing to the model when they sit in the
same basin, no matter how different their wording. They mean different things
when a hair-thin boundary runs between them, no matter how similar the
wording. The 4x4 experiments made this vivid: phrasings that a human would
call identical landed in a dozen different answer basins, and the majority
answer only held 30 percent of the variants.

This inverts the usual folklore about embedding spaces. The folk picture is
"similar meanings are nearby points." The basin picture says the interesting
structure is not distance but drainage: which way the dynamics flows from a
point, and where it pools. Two prompts can be far apart and share a meaning
(same basin), or nearly touching and mean radically different things (Wada
boundary). A geometric model of language built on proximity alone misses
everything our maps show.

## 4. Error as mis-ingression

The strongest statistical link we have: the longer a trajectory wandered
before settling, the further its answer sat from the consensus (rho 0.857 on
FPRM, settling against Hamming distance to the majority answer). Combined
with the early-FLI result, the causal reading is tempting. There exist
regions of the space from which descent is slow and the destination is far
from where the collective usually lands. Error is not a random corruption
during generation. It is a predictable consequence of where the query
ingressed, and it is visible from inside the trajectory, early, through the
local geometry (rho 0.58 between loop-3 divergence and total settling).

This is the shape of a testable claim about ordinary models: a model that
could sense its own position relative to basin boundaries would know, before
finishing, whether it is about to be wrong. Levin's cells approximate this
with chemical gradients. Our models can approximate it with neighbor
divergence. Neither needs to see the whole landscape.

## 5. The collective, from inside

4096 trajectories in violent disagreement, scattered across 1489 answers,
and the majority basin still collects 43 percent under pure greedy descent.
That robustness-under-chaos is the morphogenetic signature: the collective
outcome survives the noise of the parts. What Levin asks about cells applies
literally here. Can any single trajectory tell it is participating in a
collective decision? Yes, partially: the loop-3 divergence from its
neighbors predicts both its fate and the collective's statistics. The
geometry an agent can see locally encodes the structure of the whole it
belongs to. Whether that counts as a self-model is a question for
philosophers; that it is measurable is a question we have already answered.

## 6. What this model says language is

Putting it together: a language model is a fixed space of stabilities (the
attractor atlas), and language use is navigation of that space under boundary
conditions. The atlas is vaster than any corpus of actual conversation, the
way Levin's Platonic space is vaster than the organisms that ingress from it.
Most of its basins have never been visited by a human prompt and correspond
to no human-intended meaning; they are answers to questions nobody asked,
sitting there selectable. Training does not write the answers. Training
sculpts the drainage.

On this view, the philosophically loaded questions become empirical ones:

- Is the atlas query-independent? Same basins under paraphrase, under
  different slice seeds, under different puzzles of the same structure.
  (Partially answered: paraphrase alone moved Qwen3.6-Flash across a dozen
  basins, so the atlas is stably shaped but its basins are sensitive to
  boundary conditions in ways a naive "meaning = position" view misses.)
- Do cones scale with capability? Basins across the Qwen family, 0.6B to 8B.
  Levin's claim that collective intelligence grows by enlarging the light
  cone becomes a scaling measurement.
- Is indecision maximal at boundaries? The rigorous Wada test, beyond our
  5x5 approximation.
- Can the collective be intervened on through its parts? The saddle-kick
  experiment: perturb a stalling trajectory along its unstable manifold and
  watch the answer change. If a local touch redirects the ingress, the
  pattern/causation story has a laboratory handle.

## 7. Where this model is weakest

The analogy has known strain points, and honesty requires listing them. The
atlas is fixed per model, but real language use involves long contexts where
the landscape itself shifts token by token, so our frozen-landscape picture
is a per-decision idealization, exact for the looped models and approximate
for autoregressive ones. The basins we measure are basins of a specific
probe family; there may be structure no slice of ours touches. And "the
answer was already in the weights" needs the caveat that the weights encode
the space of possible stabilities, not the answers themselves; what ingresses
is a selection, not a dormant object waiting in a drawer. Levin's own
framework handles this with patterns being definitional rather than stored,
and our data is compatible with that reading, but the experiment does not
force it.

## 8. Status of each claim

| Claim | Status |
|---|---|
| Basins are real and measurable | done (exp01, exp04, exp06) |
| Difficulty gates fractality | done on 2 puzzles + control (exp01) |
| Settling predicts error severity | done, rho 0.857 (exp06b) |
| Local geometry predicts fate | done, rho 0.58 (exp05) |
| Approximate Wada structure | done at 64^2, 5x5 approximation (exp06); rigorous test pending |
| Light cones scale with capability | experiment designed, not run |
| Atlas is query-independent | partially; paraphrase sensitivity measured (exp07), slice-seed invariance measured (exp01 seeds 0,1) |
| Intervention through parts works | designed (saddle-kick), not run |
