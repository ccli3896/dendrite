# 04-hebbian-steering

Live site: https://ccli3896.github.io/dendrite/04-hebbian-steering/

A single self-contained page (`index.html`) with a title slide and four animations. Every simulation runs live in the browser in plain JavaScript; nothing is precomputed.

| Tab | What it shows | Model |
|---|---|---|
| 1 Reward | A 1–50–2 network learns to choose the arm that returns input to its sensory neuron | PC rule, α 0.0164, β 0.0072, η 0.01 |
| 2 Turnover | Same task while batches of hidden neurons are randomly removed or added | Same as tab 1 |
| 3 Continual | Alternating A/B context blocks; each input channel's preference for its own arm | Hebbian (PC) vs. RPE three-factor rule, 2–2 network |
| 4 Open field | An agent climbs and moves between round hills; trail turns rose near a summit | PaN model (Li et al. 2024), 1–4 network, re-tuned parameters |

Hover over a tab's title to see its full parameters.

## Controls

- Space: play / pause. R: new random seed. F: favorite the current seed. Left / right arrows: speed. 0–4: switch tab.
- Each tab has its own seed. Seeds and favorites are saved in your browser only.

## Notes

- The PC and RPE updates are ports of `ssb/models` from the paper code. With noise off and identical initial weights, the port reproduces the JAX implementation's trajectory exactly.
- The open-field demo is not the original 2024 setup. It uses round hills, a network with no hidden layer, and parameters re-tuned so agents both settle on summits and move between them.
