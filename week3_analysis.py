"""Reproducible Week 3 clustering and model evaluation exercise.

Uses only the Python standard library. Running this file writes a synthetic
customer dataset and JSON results beside the script.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from heapq import heappop, heappush
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
FEATURES = ["monthly_spend", "visit_days", "basket_items", "tenure_months"]
SEGMENTS = {
    "value_browser": ([35, 3, 1.3, 8], [14, 1.2, 0.35, 5]),
    "routine_shopper": ([85, 9, 2.1, 32], [20, 2, 0.45, 8]),
    "premium_occasional": ([185, 4, 3.8, 22], [30, 1.4, 0.6, 10]),
    "loyal_premium": ([245, 11, 4.6, 42], [30, 2, 0.5, 10]),
}


def mean(values):
    return sum(values) / len(values)


def standard_deviation(values):
    center = mean(values)
    return math.sqrt(sum((x - center) ** 2 for x in values) / len(values)) or 1.0


def euclidean_squared(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def generate_dataset():
    rng = random.Random(SEED)
    rows = []
    # 80 profiles from each known generating segment. The segment field is
    # retained only to assess clustering against known synthetic provenance.
    for segment, (center, spread) in SEGMENTS.items():
        for _ in range(80):
            values = [rng.gauss(m, s) for m, s in zip(center, spread)]
            values[0] = max(5.0, values[0])
            values[1] = min(30.0, max(1.0, values[1]))
            values[2] = max(1.0, values[2])
            values[3] = max(1.0, values[3])
            rows.append({name: value for name, value in zip(FEATURES, values)} | {"known_segment": segment})
    rng.shuffle(rows)

    centers = {name: mean([row[name] for row in rows]) for name in FEATURES}
    scales = {name: standard_deviation([row[name] for row in rows]) for name in FEATURES}
    for row in rows:
        signal = (
            0.65 * (row["monthly_spend"] - centers["monthly_spend"]) / scales["monthly_spend"]
            + 0.35 * (row["visit_days"] - centers["visit_days"]) / scales["visit_days"]
            + 0.25 * (row["basket_items"] - centers["basket_items"]) / scales["basket_items"]
            + 0.20 * (row["tenure_months"] - centers["tenure_months"]) / scales["tenure_months"]
        )
        probability = 1.0 / (1.0 + math.exp(-signal))
        row["high_value"] = int(rng.random() < probability)
    for i, row in enumerate(rows, start=1):
        row["customer_id"] = f"C{i:03d}"
    return rows


def write_dataset(rows):
    path = ROOT / "customer_segments.csv"
    columns = ["customer_id", *FEATURES, "known_segment", "high_value"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def standardize(matrix):
    means = [mean([row[j] for row in matrix]) for j in range(len(matrix[0]))]
    scales = [standard_deviation([row[j] for row in matrix]) for j in range(len(matrix[0]))]
    return [[(value - m) / s for value, m, s in zip(row, means, scales)] for row in matrix]


def kmeans(matrix, k, seed, restarts=10, max_iter=150):
    rng = random.Random(seed)
    best = None
    for _ in range(restarts):
        centers = [matrix[i][:] for i in rng.sample(range(len(matrix)), k)]
        labels = [-1] * len(matrix)
        for _iteration in range(max_iter):
            new_labels = [min(range(k), key=lambda c: euclidean_squared(row, centers[c])) for row in matrix]
            groups = [[row for row, label in zip(matrix, new_labels) if label == c] for c in range(k)]
            for c, group in enumerate(groups):
                if group:
                    centers[c] = [mean([row[j] for row in group]) for j in range(len(matrix[0]))]
                else:
                    farthest = max(range(len(matrix)), key=lambda i: euclidean_squared(matrix[i], centers[new_labels[i]]))
                    centers[c] = matrix[farthest][:]
            if new_labels == labels:
                labels = new_labels
                break
            labels = new_labels
        inertia = sum(euclidean_squared(row, centers[label]) for row, label in zip(matrix, labels))
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    return best


def silhouette_score(matrix, labels):
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    if len(groups) < 2:
        return 0.0
    distances = [[math.sqrt(euclidean_squared(a, b)) for b in matrix] for a in matrix]
    values = []
    for i, label in enumerate(labels):
        own = groups[label]
        a = sum(distances[i][j] for j in own if j != i) / max(1, len(own) - 1)
        b = min(sum(distances[i][j] for j in group) / len(group) for other, group in groups.items() if other != label)
        values.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return mean(values)


def adjusted_rand_index(actual, predicted):
    table = Counter(zip(actual, predicted))
    actual_counts = Counter(actual)
    predicted_counts = Counter(predicted)
    choose2 = lambda n: n * (n - 1) / 2
    total = choose2(len(actual))
    if not total:
        return 1.0
    index = sum(choose2(count) for count in table.values())
    expected = sum(choose2(n) for n in actual_counts.values()) * sum(choose2(n) for n in predicted_counts.values()) / total
    maximum = 0.5 * (sum(choose2(n) for n in actual_counts.values()) + sum(choose2(n) for n in predicted_counts.values()))
    return (index - expected) / (maximum - expected) if maximum != expected else 1.0


def ward_labels(matrix, k):
    """Ward agglomeration with a heap; returns labels for a requested k."""
    n = len(matrix)
    centroids = {i: matrix[i][:] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    members = {i: [i] for i in range(n)}
    active = set(range(n))
    heap = []

    def cost(a, b):
        na, nb = sizes[a], sizes[b]
        return na * nb / (na + nb) * euclidean_squared(centroids[a], centroids[b])

    for i in range(n):
        for j in range(i + 1, n):
            heappush(heap, (cost(i, j), i, j))
    next_id = n
    while len(active) > k:
        while True:
            _distance, a, b = heappop(heap)
            if a in active and b in active:
                break
        size = sizes[a] + sizes[b]
        centroid = [(sizes[a] * x + sizes[b] * y) / size for x, y in zip(centroids[a], centroids[b])]
        merged_members = members[a] + members[b]
        active.remove(a)
        active.remove(b)
        active.add(next_id)
        centroids[next_id], sizes[next_id], members[next_id] = centroid, size, merged_members
        for other in active:
            if other != next_id:
                heappush(heap, (cost(next_id, other), min(next_id, other), max(next_id, other)))
        next_id += 1
    labels = [None] * n
    for label, cluster_id in enumerate(sorted(active)):
        for i in members[cluster_id]:
            labels[i] = label
    return labels


def pca_two_components(matrix):
    """Power iteration with deflation for the first two covariance eigenpairs."""
    n, d = len(matrix), len(matrix[0])
    covariance = [[sum(row[i] * row[j] for row in matrix) / (n - 1) for j in range(d)] for i in range(d)]
    residual = [row[:] for row in covariance]
    eigenvalues, vectors = [], []
    for component in range(2):
        vector = [1.0 if i == component else 0.25 for i in range(d)]
        norm = math.sqrt(sum(x * x for x in vector))
        vector = [x / norm for x in vector]
        for _ in range(1000):
            updated = [sum(residual[i][j] * vector[j] for j in range(d)) for i in range(d)]
            norm = math.sqrt(sum(x * x for x in updated))
            updated = [x / norm for x in updated]
            if sum((a - b) ** 2 for a, b in zip(updated, vector)) < 1e-20:
                vector = updated
                break
            vector = updated
        eigenvalue = sum(vector[i] * residual[i][j] * vector[j] for i in range(d) for j in range(d))
        eigenvalues.append(eigenvalue)
        vectors.append(vector)
        for i in range(d):
            for j in range(d):
                residual[i][j] -= eigenvalue * vector[i] * vector[j]
    total_variance = sum(covariance[i][i] for i in range(d))
    return eigenvalues, [v / total_variance for v in eigenvalues], vectors


def stratified_holdout(labels, fraction=0.25, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)
    test = []
    for indices in by_class.values():
        rng.shuffle(indices)
        test.extend(indices[:round(len(indices) * fraction)])
    test_set = set(test)
    train = [i for i in range(len(labels)) if i not in test_set]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def stratified_folds(labels, indices, folds=5, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i in indices:
        by_class[labels[i]].append(i)
    result = [[] for _ in range(folds)]
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        for position, i in enumerate(class_indices):
            result[position % folds].append(i)
    return result


def fit_logistic(matrix, labels, train_indices, c_value, iterations=1800):
    means = [mean([matrix[i][j] for i in train_indices]) for j in range(len(matrix[0]))]
    scales = [standard_deviation([matrix[i][j] for i in train_indices]) for j in range(len(matrix[0]))]
    x = [[(matrix[i][j] - means[j]) / scales[j] for j in range(len(matrix[0]))] for i in train_indices]
    y = [labels[i] for i in train_indices]
    weights = [0.0] * len(matrix[0])
    bias = 0.0
    regularization = 1.0 / c_value
    rate = 0.45 / (1.0 + regularization + len(weights) * 0.25)
    n = len(y)
    for _ in range(iterations):
        grad_w = [regularization * weight for weight in weights]
        grad_b = 0.0
        for row, target in zip(x, y):
            z = max(-35.0, min(35.0, bias + sum(a * b for a, b in zip(weights, row))))
            probability = 1.0 / (1.0 + math.exp(-z))
            error = probability - target
            grad_b += error
            for j, value in enumerate(row):
                grad_w[j] += error * value
        bias -= rate * grad_b / n
        weights = [weight - rate * gradient / n for weight, gradient in zip(weights, grad_w)]
    return (means, scales, weights, bias)


def predict_probability(model, row):
    means, scales, weights, bias = model
    z = bias + sum(weight * (value - m) / s for value, m, s, weight in zip(row, means, scales, weights))
    z = max(-35.0, min(35.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def f1_score(actual, predicted):
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def roc_auc(actual, scores):
    positives = sum(actual)
    negatives = len(actual) - positives
    if not positives or not negatives:
        return 0.5
    ordered = sorted(zip(scores, actual))
    rank_sum = 0.0
    position = 1
    while position <= len(ordered):
        end = position
        while end < len(ordered) and ordered[end][0] == ordered[position - 1][0]:
            end += 1
        average_rank = (position + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[position - 1:end])
        position = end + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(actual, scores):
    predicted = [int(score >= 0.5) for score in scores]
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": (tp + tn) / len(actual),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": f1_score(actual, predicted),
        "roc_auc": roc_auc(actual, scores),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def evaluate(rows):
    raw = [[row[name] for name in FEATURES] for row in rows]
    scaled = standardize(raw)
    actual_segments = [row["known_segment"] for row in rows]
    cluster_table = []
    selected_kmeans = None
    for k in range(2, 7):
        inertia, labels = kmeans(scaled, k, SEED + k)
        result = {"k": k, "inertia": inertia, "silhouette": silhouette_score(scaled, labels), "adjusted_rand_index": adjusted_rand_index(actual_segments, labels)}
        cluster_table.append(result)
        if k == 4:
            selected_kmeans = labels
    ward = ward_labels(scaled, 4)
    eigens, ratios, _vectors = pca_two_components(scaled)
    targets = [row["high_value"] for row in rows]
    train, test = stratified_holdout(targets)
    folds = stratified_folds(targets, train, folds=5)
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_table = []
    for c_value in c_grid:
        fold_f1, fold_auc = [], []
        for fold_number, validation in enumerate(folds):
            validation_set = set(validation)
            fit_indices = [i for i in train if i not in validation_set]
            model = fit_logistic(raw, targets, fit_indices, c_value)
            scores = [predict_probability(model, raw[i]) for i in validation]
            truth = [targets[i] for i in validation]
            fold_f1.append(f1_score(truth, [int(score >= 0.5) for score in scores]))
            fold_auc.append(roc_auc(truth, scores))
        cv_table.append({"C": c_value, "mean_f1": mean(fold_f1), "mean_roc_auc": mean(fold_auc)})
    best = max(cv_table, key=lambda result: (result["mean_f1"], result["mean_roc_auc"], -result["C"]))
    model = fit_logistic(raw, targets, train, best["C"])
    test_scores = [predict_probability(model, raw[i]) for i in test]
    test_targets = [targets[i] for i in test]
    test_metrics = classification_metrics(test_targets, test_scores)
    return {
        "seed": SEED,
        "rows": len(rows),
        "features": FEATURES,
        "target": "high_value",
        "positive_rate": mean(targets),
        "holdout": {"train_rows": len(train), "test_rows": len(test), "test_positive_rate": mean(test_targets)},
        "clustering": {
            "kmeans": cluster_table,
            "selected_k": 4,
            "selected_silhouette": next(row["silhouette"] for row in cluster_table if row["k"] == 4),
            "selected_adjusted_rand_index": adjusted_rand_index(actual_segments, selected_kmeans),
            "ward_k": 4,
            "ward_silhouette": silhouette_score(scaled, ward),
            "ward_adjusted_rand_index": adjusted_rand_index(actual_segments, ward),
        },
        "pca": {"components": 2, "explained_variance_ratios": ratios, "cumulative_explained_variance": sum(ratios)},
        "cross_validation": {"folds": 5, "scoring": "mean F1 at threshold 0.5; ROC-AUC tracked", "grid": cv_table, "selected_C": best["C"]},
        "test_metrics": test_metrics,
    }


def main():
    rows = generate_dataset()
    write_dataset(rows)
    results = evaluate(rows)
    output = ROOT / "week3_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
