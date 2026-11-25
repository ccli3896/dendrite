# 04-hebbian-steering

Live site: https://ccli3896.github.io/dendrite/04-hebbian-steering/

A single self-contained page (`index.html`) with a title slide, five demos and two attractor views. Every simulation runs live in the browser in plain JavaScript; nothing is precomputed. On a laptop the page is a 16:10 stage with the picture and the plot side by side. On phones, portrait tablets and short screens it switches to one scrolling column, with the controls at the top and the tab bar scrolling sideways.

Most tabs let you switch between seven local learning rules: PC (prospective configuration), BCM, Oja, Law & Cooper (LC), covariance (COV), latent predictive learning (LPL) and an RPE-gated three-factor rule.

| Tab | What it shows | Model |
|---|---|---|
| 1 Passive | Same input to a 2-neuron network under each rule; 3D trajectory of (x₀, x₁, w) plus time series. Tick several rules to overlay them | 2-neuron network (main Fig. 1a); s = 1 for 500 steps, then 0; all rules share the same noise |
| 2 Reward | A 1–50–2 network learns to choose the arm that returns input to its sensory neuron | Pick a rule; each runs at its large-network reward-seeking operating point (Supp. Table, main Fig. 1f); mean of 16 networks |
| 3 Turnover | Same task while batches of hidden neurons are randomly removed or added | PC rule, α 0.0164, β 0.0072, η 0.01 |
| 4 Continual | Alternating A/B context blocks; each input channel's preference for its own arm | Hebbian (PC) vs. RPE three-factor rule, 2–2 network |
| 5 Open field | An agent climbs and moves between round hills; trail turns rose near a summit | PaN model (Li et al. 2024), 1–4 network, re-tuned parameters |
| 6 Deterministic (attractors) | World-in-the-loop dynamics in (x_in, W_diff, X_diff) space, with choice raster and P(choose a) | 1–2 network (main Fig. 1i,j); fixed points drawn for PC |
| 7 Matching pennies (attractors) | Alternating 20k-step phases against a pattern-predicting computer opponent and a coin flip; choices and 3-bit choice entropy | 1–2 network, each rule at its Fig. 3h–k cell |

Each tab has a short blurb next to its title describing what is being simulated. Hover over the title to see the full parameters. On the 3D plots, drag to rotate and double-click to reset the view.

## Controls

- Space: play / pause. R: new random seed. F: favorite the current seed. Left / right arrows: speed. 0–7: switch tab.
- Each tab has its own seed. Seeds and favorites are saved in your browser only.

## Notes

- The learning-rule updates are ports of `ssb/models` from the paper code. With noise off and identical initial weights, the PC port reproduces the JAX implementation's trajectory exactly.
- The matching-pennies opponent is a port of `ssb/envs/matchingpennies.py`: it looks up the agent's last 3 choices in a 100-choice history and predicts the next one when a z-test is significant (p < 0.05), otherwise it guesses at random.
- The open-field demo is not the original 2024 setup. It uses round hills, a network with no hidden layer, and parameters re-tuned so agents both settle on summits and move between them.
