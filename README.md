# Wait-Duration Features for Predicting Capacity Sensitivity of Schedules

This experiment is motivated by "Learning sensitivity of schedules by analyzing the search process", which proposes classifiers trained on features collected from a Cumulative constraint during search, such as how often a task waits after a resource is released. We ask whether adding wait-duration features and a class-balanced, cross-validation-tuned gradient-boosting model can better predict whether one extra unit of resource capacity reduces makespan than a random forest that sees only wait counts.

**Question:** Can wait-duration features with a class-balanced, CV-tuned gradient-boosting model predict whether one extra unit of resource capacity reduces makespan better than a random forest on count-only wait features?

**Dataset:** Simulated single-cumulative-resource scheduling instances, generated in `analysis.py` with numpy (random durations, demands, and precedence chains). No download is needed. This is not the paper's own dataset.

**Baseline:** Random forest on count-only wait features (how often each task waits through resource releases, plus instance size and capacity).

**Improved:** Class-balanced LightGBM, with hyper-parameters chosen by cross-validation on the training split only, using the same counts plus the total, mean, and maximum wait durations.

**How to run:**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json` (and an optional chart to `figures/`).

## Results (latest run)

On a simulated single-resource scheduling dataset (6,000 instances, 1,800 held out for test), ROC-AUC rose from 0.622 for a standard random forest on count-based wait features to 0.704 for class-balanced LightGBM with CV-tuned hyperparameters and wait-duration features, though F1 fell from 0.869 to 0.770.

| Metric | Value |
|---|---|
| baseline_score | 0.6217 |
| baseline_score_std | 0.001356 |
| improved_score | 0.7044 |
| improved_score_std | 0 |
| improvement | 0.08269 |
| baseline_f1 | 0.8689 |
| improved_f1 | 0.7698 |
| baseline_balanced_accuracy | 0.5504 |
| improved_balanced_accuracy | 0.6496 |
| improved_cv_auc | 0.6839 |
| positive_rate | 0.803 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Labels come from a simple serial SGS heuristic on random instances, not from the paper's CP solver search; results describe this simulation only. The paper's reported result is not stated. Test set used once; tuning used training-set CV only. Both models use the same seeds (42, 43, 44) and the same evaluation function.

## Reference

This project is motivated by the following paper:

> MA Ménard, CG Quimper, J Gaudreault. *Learning sensitivity of schedules by analyzing the search process*. cp2019.a4cp.org. https://cp2019.a4cp.org/accepted_dp/08-marc_andre_menard.pdf

Paper author: Prof. Claude-Guy Quimper, Université Laval (claude-guy.quimper@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
