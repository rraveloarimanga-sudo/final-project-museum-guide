# Git workflow (group)

## Branches
- `main` — always runs; only merged via Pull/Merge Request reviewed by one teammate.
- One branch per step: `feat/data-eda`, `feat/baseline`, `feat/mlp`, `feat/diagnosis`,
  `feat/resnet`, `feat/search`, `feat/benchmark`, `docs/slides`.

## Commit convention
`<type>(<scope>): <what, in the imperative>` — types: `feat`, `fix`, `exp`, `docs`, `refactor`, `chore`.
`exp` = an experiment whose result is saved in `outputs/` (mention the key number in the message).

## Suggested history
```
chore: init repo structure, requirements and gitignore
feat(data): add loader, label encoding and stratified 70/15/15 split
feat(eda): plot class balance, sample grid, majority-class floor
feat(baseline): logistic regression on colour histograms + HOG
feat(mlp): feed-forward net from nn.Linear with manual training loop
exp(mlp): baseline MLP run — val acc XX%, overfits after epoch N
exp(diagnosis): learning-rate sweep shows divergence above lr=...
exp(diagnosis): gradient-norm logging reveals vanishing grads with sigmoid
fix(mlp): add dropout + weight decay + early stopping — val gap XX -> YY
feat(resnet): ResNet18 frozen features + linear head
exp(resnet): fine-tune layer4 — val acc XX%
feat(search): text-to-artwork retrieval over descriptions
exp(benchmark): CPU latency and model size for kiosk recommendation
docs: experiment log and README results table
```

## Split of work (example for 4)
| Person | Owns | Presents |
|---|---|---|
| A | data, EDA, baseline | Problem & data (2 min) |
| B | MLP + training loop | MLP & monitoring (2.5 min) |
| C | diagnosis experiments | Failures & fixes (2.5 min) |
| D | ResNet, search, benchmark | ResNet, search, recommendation (3 min) |

Everyone must be able to answer questions on every part (individual deliberation).
