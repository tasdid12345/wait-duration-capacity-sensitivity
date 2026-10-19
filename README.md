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

## Reference

This project is motivated by the following paper:

> MA Ménard, CG Quimper, J Gaudreault. *Learning sensitivity of schedules by analyzing the search process*. cp2019.a4cp.org. https://cp2019.a4cp.org/accepted_dp/08-marc_andre_menard.pdf

Paper author: Prof. Claude-Guy Quimper, Université Laval (claude-guy.quimper@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
