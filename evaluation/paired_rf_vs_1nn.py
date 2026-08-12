from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["ProductType", "ProductGroup", "ProductFamily", "ProductLine", "ProductKey"]
PREFIXES = {target: "product_" + target.removeprefix("Product").lower() for target in TARGETS}
FEATURES = ["GPCBrickCode", "UNSPSCNumber", "ProductDescription"]
RANDOM_STATE = 42
MIN_CLASS_COUNT = 10
TOP_K = 3
RETRIEVAL_WINDOW = 50


def wilson_interval(correct, total):
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - margin, center + margin


def paired_difference_interval(rf_correct, nn_correct, repeats=20_000):
    differences = rf_correct.astype(np.int8) - nn_correct.astype(np.int8)
    rng = np.random.default_rng(RANDOM_STATE)
    estimates = np.empty(repeats)
    for start in range(0, repeats, 1_000):
        size = min(1_000, repeats - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        estimates[start : start + size] = differences[indices].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975])


def first_distinct_labels(labels, similarities, k=TOP_K, window=RETRIEVAL_WINDOW):
    nearest = np.argpartition(-similarities, kth=window - 1, axis=1)[:, :window]
    ranked = nearest[
        np.arange(len(nearest))[:, None],
        np.argsort(-similarities[np.arange(len(nearest))[:, None], nearest], axis=1),
    ]
    predictions = []
    for row in ranked:
        candidates = []
        for label in labels[row]:
            if label not in candidates:
                candidates.append(label)
                if len(candidates) == k:
                    break
        predictions.append(candidates)
    return predictions


def evaluate_hits(true_labels, rf_probabilities, rf_classes, nn_labels, similarities):
    rf_order = np.argsort(rf_probabilities, axis=1)[:, ::-1]
    rf_top1 = rf_classes[rf_order[:, 0]] == true_labels
    rf_top3 = np.any(rf_classes[rf_order[:, :TOP_K]] == true_labels[:, None], axis=1)

    nn_order = similarities.argmax(axis=1)
    nn_top1 = nn_labels[nn_order] == true_labels
    nn_candidates = first_distinct_labels(nn_labels, similarities)
    nn_top3 = np.array([truth in candidates for truth, candidates in zip(true_labels, nn_candidates)])
    return {"Top-1": (rf_top1, nn_top1), "Top-3": (rf_top3, nn_top3)}


def summarize(target, metric, rf_correct, nn_correct):
    total = len(rf_correct)
    rf_wins = int(np.sum(rf_correct & ~nn_correct))
    nn_wins = int(np.sum(~rf_correct & nn_correct))
    both_correct = int(np.sum(rf_correct & nn_correct))
    both_wrong = int(np.sum(~rf_correct & ~nn_correct))
    discordant = rf_wins + nn_wins
    p_value = binomtest(min(rf_wins, nn_wins), discordant, 0.5).pvalue if discordant else 1.0
    rf_count = int(rf_correct.sum())
    nn_count = int(nn_correct.sum())
    rf_ci = wilson_interval(rf_count, total)
    nn_ci = wilson_interval(nn_count, total)
    difference_ci = paired_difference_interval(rf_correct, nn_correct)
    return {
        "Target": target,
        "Metric": metric,
        "TestRows": total,
        "RF_Correct": rf_count,
        "RF_Accuracy": rf_count / total,
        "RF_CI_Low": rf_ci[0],
        "RF_CI_High": rf_ci[1],
        "NN_Correct": nn_count,
        "NN_Accuracy": nn_count / total,
        "NN_CI_Low": nn_ci[0],
        "NN_CI_High": nn_ci[1],
        "Difference": (rf_count - nn_count) / total,
        "Difference_CI_Low": difference_ci[0],
        "Difference_CI_High": difference_ci[1],
        "BothCorrect": both_correct,
        "RFOnlyCorrect": rf_wins,
        "NNOnlyCorrect": nn_wins,
        "BothWrong": both_wrong,
        "McNemar_P": p_value,
    }


def main():
    data = pd.read_excel(ROOT / "datasets/product_classified_Full.xlsx", sheet_name="in")
    data = data[FEATURES + TARGETS].copy()
    data.dropna(subset=FEATURES, how="all", inplace=True)

    raw_text = data["ProductDescription"].fillna("")
    combined_text = (
        "ProductDescription: " + raw_text
        + " | GPCBrickCode: " + data["GPCBrickCode"].fillna("Unknown_GPCBrickCode").astype(str)
        + " | UNSPSCNumber: " + data["UNSPSCNumber"].fillna("Unknown_UNSPSCNumber").astype(str)
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    nn_embeddings = embedder.encode(
        combined_text.tolist(),
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    rf_text = raw_text.str.strip()
    text_codes, unique_text = pd.factorize(rf_text[rf_text != ""], sort=False)
    unique_embeddings = embedder.encode(
        unique_text.tolist(), batch_size=64, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    rf_embeddings = np.zeros((len(data), unique_embeddings.shape[1]), dtype=np.float32)
    rf_embeddings[np.flatnonzero((rf_text != "").to_numpy())] = unique_embeddings[text_codes]

    data["GPCBrickCode"] = data["GPCBrickCode"].fillna("Missing")
    data["UNSPSCNumber"] = data["UNSPSCNumber"].fillna(-1).astype(int).astype(str)
    results = []

    for target in TARGETS:
        counts = data[target].value_counts()
        labels = data[target].where(~data[target].isin(counts[counts < MIN_CLASS_COUNT].index), "Other")
        train_positions, test_positions = train_test_split(
            np.arange(len(data)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        prefix = PREFIXES[target]
        model = joblib.load(ROOT / "artifacts" / f"{prefix}_rf_3feat_model.joblib")
        encoder = joblib.load(ROOT / "artifacts" / f"{prefix}_rf_3feat_ohe.joblib")
        categorical = encoder.transform(data.iloc[test_positions][["GPCBrickCode", "UNSPSCNumber"]])
        rf_features = np.hstack([categorical, rf_embeddings[test_positions]])
        rf_probabilities = model.predict_proba(rf_features)

        similarities = nn_embeddings[test_positions] @ nn_embeddings[train_positions].T
        train_labels = labels.iloc[train_positions].to_numpy()
        test_labels = labels.iloc[test_positions].to_numpy()
        hits = evaluate_hits(
            test_labels,
            rf_probabilities,
            model.classes_,
            train_labels,
            similarities,
        )
        for metric, (rf_correct, nn_correct) in hits.items():
            results.append(summarize(target, metric, rf_correct, nn_correct))
        del model, encoder, categorical, rf_features, rf_probabilities, similarities

    output = ROOT / "evaluation"
    output.mkdir(exist_ok=True)
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output / "paired_rf_vs_1nn_results.csv", index=False)
    results_frame.to_json(output / "paired_rf_vs_1nn_results.json", orient="records", indent=2)
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()