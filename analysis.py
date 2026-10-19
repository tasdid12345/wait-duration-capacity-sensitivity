"""Can wait-duration features + class-balanced, CV-tuned LightGBM predict whether
one extra unit of resource capacity reduces makespan better than a random forest
on count-only wait features?  Data: simulated single-cumulative-resource instances."""
import json
import math
import os
import sys

import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]           # seeds for the randomized models
N_INST = 6000                  # number of simulated instances
N_MAX = 12                     # max tasks per instance (count features padded to this)
CAP_RANGE = (3, 8)             # capacity C drawn from [3, 8)


def serial_sgs(dur, dem, prec, key, cap):
    """Serial schedule-generation scheme with a single cumulative resource.
    Returns makespan, per-task wait counts (# resource releases waited through)
    and per-task wait durations (start minus precedence-earliest start)."""
    n = len(dur)
    usage = np.zeros(int(dur.sum()) + 2, dtype=int)   # resource profile over time
    finish = np.zeros(n, dtype=int)
    scheduled = np.zeros(n, dtype=bool)
    wait_cnt = np.zeros(n, dtype=int)
    wait_dur = np.zeros(n, dtype=int)
    for _ in range(n):
        # eligible = all predecessors already scheduled; pick the lowest random key
        elig = [j for j in range(n) if not scheduled[j] and all(scheduled[p] for p in prec[j])]
        j = min(elig, key=lambda k: key[k])
        d = dur[j]
        est = max([finish[p] for p in prec[j]], default=0)   # precedence-earliest start
        # resource usage only drops at finish times, so feasible starts are est or a finish time > est
        cands = sorted({est} | {int(f) for f in finish[scheduled] if f > est})
        fails = 0
        for t in cands:
            if np.all(usage[t:t + d] + dem[j] <= cap):
                break
            fails += 1                                   # waited through a release event
        usage[t:t + d] += dem[j]
        finish[j] = t + d
        scheduled[j] = True
        wait_cnt[j] = fails
        wait_dur[j] = t - est
    return finish.max(), wait_cnt, wait_dur


def make_instance(rng):
    """Random single-resource instance: durations, demands <= C, random precedence edges."""
    n = int(rng.integers(6, N_MAX + 1))
    cap = int(rng.integers(*CAP_RANGE))
    dur = rng.integers(1, 11, n)
    dem = rng.integers(1, cap + 1, n)
    prec = [[] for _ in range(n)]
    for j in range(1, n):                                # sparse random precedence chains
        if rng.random() < 0.3:
            prec[j].append(int(rng.integers(0, j)))
    key = rng.random(n)                                  # random priority list
    return n, cap, dur, dem, prec, key


def evaluate(y, proba):
    """Shared evaluation: ROC-AUC (main) plus balanced accuracy and positive-class F1."""
    pred = (proba >= 0.5).astype(int)
    return {"auc": roc_auc_score(y, proba),
            "bal_acc": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0)}


# ---- 1. simulate data (same loop for everything) ----
print("Simulating instances...")
rng = np.random.default_rng(SEED)
Xb, Xi, y = [], [], []
for _ in range(N_INST):
    n, cap, dur, dem, prec, key = make_instance(rng)
    ms_c, cnt, wdur = serial_sgs(dur, dem, prec, key, cap)
    ms_c1, _, _ = serial_sgs(dur, dem, prec, key, cap + 1)
    counts = np.pad(np.sort(cnt)[::-1], (0, N_MAX - n))  # paper-style wait counts
    base = np.concatenate([[n, cap], counts])
    waits = [wdur.sum(), wdur.mean(), wdur.max()]        # total / mean / max wait durations
    Xb.append(base)
    Xi.append(np.concatenate([base, waits]))
    y.append(int(ms_c1 < ms_c))                          # 1 = extra unit reduces makespan
Xb, Xi, y = np.array(Xb, float), np.array(Xi, float), np.array(y)
if y.min() == y.max() or y.sum() < 20 or (1 - y).sum() < 20:
    sys.exit("Label has too few examples of one class; cannot produce meaningful metrics.")
print(f"Instances: {len(y)}, positive rate: {y.mean():.3f}")

# ---- 2. single stratified train/test split (fixed seed) ----
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
ytr, yte = y[idx_tr], y[idx_te]

# ---- 3. baseline: random forest on count-only features ----
print("Baseline: random forest (counts only)...")
base_res = []
for s in SEEDS:
    rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
    rf.fit(Xb[idx_tr], ytr)
    base_res.append(evaluate(yte, rf.predict_proba(Xb[idx_te])[:, 1]))

# ---- 4. improved: class-balanced LightGBM, CV-tuned on training data only ----
print("Improved: CV-tuning class-balanced LightGBM on training set...")
grid = GridSearchCV(
    lgb.LGBMClassifier(class_weight="balanced", n_estimators=300, random_state=SEED, verbose=-1),
    param_grid={"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "min_child_samples": [20, 50]},
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=-1)
grid.fit(Xi[idx_tr], ytr)
print(f"  best CV AUC: {grid.best_score_:.4f}, params: {grid.best_params_}")
imp_res = []
for s in SEEDS:
    m = lgb.LGBMClassifier(class_weight="balanced", n_estimators=300, random_state=s,
                           verbose=-1, **grid.best_params_)
    m.fit(Xi[idx_tr], ytr)
    imp_res.append(evaluate(yte, m.predict_proba(Xi[idx_te])[:, 1]))


def agg(res, k):
    """Mean and std of one metric across seeds."""
    v = np.array([r[k] for r in res], float)
    return float(v.mean()), float(v.std())


b_auc, b_auc_sd = agg(base_res, "auc")
i_auc, i_auc_sd = agg(imp_res, "auc")
if not all(math.isfinite(v) for v in (b_auc, i_auc)):
    sys.exit("Non-finite ROC-AUC; cannot produce meaningful metrics.")

metrics = {
    "baseline_score": b_auc, "baseline_score_std": b_auc_sd,
    "improved_score": i_auc, "improved_score_std": i_auc_sd,
    "improvement": i_auc - b_auc,
    "baseline_f1": agg(base_res, "f1")[0], "improved_f1": agg(imp_res, "f1")[0],
    "baseline_balanced_accuracy": agg(base_res, "bal_acc")[0],
    "improved_balanced_accuracy": agg(imp_res, "bal_acc")[0],
    "improved_cv_auc": float(grid.best_score_),
    "positive_rate": float(y.mean()),
}
print(json.dumps(metrics, indent=2))

# ---- 5. optional chart ----
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar(["baseline RF", "improved LGBM"], [b_auc, i_auc], yerr=[b_auc_sd, i_auc_sd],
       color=["grey", "steelblue"], capsize=5)
ax.set_ylabel("test ROC-AUC (mean ± std over seeds)")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/auc_comparison.png", dpi=120)

# ---- 6. write results ----
results = {
    "question": "Can wait-duration features with a class-balanced, CV-tuned gradient-boosting model "
                "predict whether one extra unit of resource capacity reduces makespan better than a "
                "random forest on count-only wait features?",
    "method": "Simulated single-resource scheduling instances (numpy); serial schedule-generation at "
              "capacity C and C+1; label = makespan drops. Stratified 70/30 split, fixed seed.",
    "dataset": {
        "name": "Simulated single-cumulative-resource scheduling instances (numpy, generated at run time)",
        "source_url": "https://numpy.org/",
        "n_rows": int(len(y)),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "ROC-AUC",
    "higher_is_better": True,
    "baseline": "Random forest (300 trees) on count-only wait features: instance size, capacity, "
                "and sorted per-task counts of resource releases waited through.",
    "improved": "Class-balanced LightGBM with hyper-parameters chosen by cross-validation on the "
                "training set, using the same counts plus total, mean, and maximum wait durations.",
    "metrics": {k: float(v) for k, v in metrics.items()},
    "notes": "Labels come from a simple serial SGS heuristic on random instances, not from the "
             "paper's CP solver search; results describe this simulation only. The paper's reported "
             "result is not stated. Test set used once; tuning used training-set CV only. Both models "
             "use the same seeds (42, 43, 44) and the same evaluation function.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote results.json")
