import argparse
import json
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from scripts.generate_synthetic_data import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("local_mode")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_data(data_dir: Path, num_users: int, num_items: int, num_interactions: int, seed: int) -> Dict[str, pd.DataFrame]:
    generator = SyntheticDataGenerator(seed=seed)
    logger.info("Generating synthetic dataset (users=%s, items=%s, interactions=%s)", num_users, num_items, num_interactions)

    users_df = generator.generate_users(num_users)
    items_df = generator.generate_items(num_items)
    interactions_df = generator.generate_interactions(users_df, items_df, num_interactions)

    users_path = data_dir / "users.csv"
    items_path = data_dir / "items.csv"
    interactions_path = data_dir / "interactions.csv"

    users_df.to_csv(users_path, index=False)
    items_df.to_csv(items_path, index=False)
    interactions_df.to_csv(interactions_path, index=False)

    logger.info("Synthetic data written to %s", data_dir)
    return {
        "users_path": users_path,
        "items_path": items_path,
        "interactions_path": interactions_path,
        "users_df": users_df,
        "items_df": items_df,
        "interactions_df": interactions_df,
    }


def train_models(args: argparse.Namespace):
    data_dir = ensure_directory(Path(args.data_dir))
    artifact_dir = ensure_directory(Path(args.artifact_dir))

    start_time = time.perf_counter()
    dataset = generate_data(
        data_dir,
        num_users=args.num_users,
        num_items=args.num_items,
        num_interactions=args.num_interactions,
        seed=args.seed,
    )
    data_generation_secs = time.perf_counter() - start_time
    logger.info("Data generation finished in %.2fs", data_generation_secs)

    tf_trainer_cls = None
    xgb_trainer_cls = None

    try:
        from pipelines.training.tf_recommenders_trainer import TFRecommendersTrainer
        tf_trainer_cls = TFRecommendersTrainer
    except Exception as exc:  # pragma: no cover
        logger.warning("TF Recommenders trainer unavailable: %s", exc)

    try:
        from pipelines.training.xgboost_trainer import XGBoostRankingTrainer
        xgb_trainer_cls = XGBoostRankingTrainer
    except Exception as exc:  # pragma: no cover
        logger.warning("XGBoost trainer unavailable: %s", exc)

    tf_duration = 0.0
    xgb_duration = 0.0
    tf_ran = False
    xgb_ran = False

    if tf_trainer_cls:
        tf_dir = ensure_directory(artifact_dir / "tf_recommenders")
        tf_trainer = tf_trainer_cls(output_dir=tf_dir, embedding_dim=args.embedding_dim, learning_rate=args.learning_rate)
        tf_start = time.perf_counter()
        try:
            tf_trainer.train(
                interactions_path=str(dataset["interactions_path"]),
                users_path=str(dataset["users_path"]),
                items_path=str(dataset["items_path"]),
                epochs=args.epochs,
                validation_split=args.validation_split,
            )
            tf_duration = time.perf_counter() - tf_start
            tf_ran = True
            logger.info("TF Recommenders training finished in %.2fs", tf_duration)
        except Exception as exc:
            tf_duration = time.perf_counter() - tf_start
            logger.warning("TF Recommenders training failed after %.2fs: %s", tf_duration, exc)
    else:
        logger.info("TF Recommenders training skipped because the trainer import failed")

    if xgb_trainer_cls:
        xgb_dir = ensure_directory(artifact_dir / "xgboost")
        xgb_trainer = xgb_trainer_cls(output_dir=xgb_dir, objective=args.xgb_objective)
        xgb_start = time.perf_counter()
        try:
            xgb_trainer.train(
                interactions_path=str(dataset["interactions_path"]),
                users_path=str(dataset["users_path"]),
                items_path=str(dataset["items_path"]),
                validation_split=args.validation_split,
                model_params={
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.xgb_learning_rate,
                },
            )
            xgb_duration = time.perf_counter() - xgb_start
            xgb_ran = True
            logger.info("XGBoost training finished in %.2fs", xgb_duration)
        except Exception as exc:
            xgb_duration = time.perf_counter() - xgb_start
            logger.warning("XGBoost training failed after %.2fs: %s", xgb_duration, exc)
    else:
        logger.info("XGBoost training skipped because the trainer import failed")

    pipeline_runtime = time.perf_counter() - start_time
    summary = {
        "pipeline_runtime_secs": round(pipeline_runtime, 2),
        "data_generation_secs": round(data_generation_secs, 2),
        "tf_training_secs": round(tf_duration, 2),
        "xgb_training_secs": round(xgb_duration, 2),
        "tf_training_executed": tf_ran,
        "xgb_training_executed": xgb_ran,
        "data_dir": str(data_dir),
        "artifact_dir": str(artifact_dir),
    }

    summary_path = artifact_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Training summary written to %s", summary_path)


def compute_base_scores(items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Dict[str, float]:
    popularity = interactions_df.groupby("item_id")["rating"].count().to_dict()
    max_popularity = max(popularity.values()) if popularity else 1

    base_scores: Dict[str, float] = {}
    for _, row in items_df.iterrows():
        item_id = row["item_id"]
        base_scores[item_id] = (
            (popularity.get(item_id, 0) / max_popularity) * 0.5
            + row.get("popularity_score", 0) * 0.3
            + row.get("quality_score", 0) * 0.2
        )

    return base_scores


def split_interactions(interactions_df: pd.DataFrame, test_ratio: float) -> pd.DataFrame:
    if "timestamp" in interactions_df.columns:
        interactions_df = interactions_df.sort_values("timestamp")
    test_size = max(1, int(len(interactions_df) * test_ratio))
    return interactions_df.iloc[:-test_size], interactions_df.iloc[-test_size:]


def calculate_recall_at_k(recommendations: Dict[str, List[str]], test_df: pd.DataFrame, k: int) -> float:
    test_groups = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    total_recall = 0
    counted_users = 0
    for user_id, rec_items in recommendations.items():
        actual_items = test_groups.get(user_id, set())
        if not actual_items:
            continue
        hits = len(set(rec_items[:k]) & actual_items)
        total_recall += hits / len(actual_items)
        counted_users += 1
    return total_recall / counted_users if counted_users else 0


def calculate_coverage(recommendations: Dict[str, List[str]], total_items: int) -> float:
    unique_items = {item for user_recs in recommendations.values() for item in user_recs}
    return len(unique_items) / total_items if total_items else 0


def calculate_diversity(recommendations: Dict[str, List[str]], items_df: pd.DataFrame) -> float:
    genre_map = items_df.set_index("item_id")["genre"].to_dict()
    diversities = []
    for user_recs in recommendations.values():
        genres = {genre_map.get(item) for item in user_recs if genre_map.get(item)}
        if user_recs:
            diversities.append(len(genres) / len(user_recs))
    return statistics.mean(diversities) if diversities else 0


def build_recommendations(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    base_scores: Dict[str, float],
    user_histories: Dict[str, Set[str]],
    k: int,
) -> Dict[str, List[str]]:
    recommendations: Dict[str, List[str]] = {}

    for _, user in users_df.iterrows():
        user_id = user["user_id"]
        preferred_genres = set(json.loads(user["preferred_genres"]))
        user_history = user_histories.get(user_id, set())

        scored_items = []
        for _, item in items_df.iterrows():
            item_id = item["item_id"]
            if item_id in user_history:
                continue
            match_score = 0.3 if item["genre"] in preferred_genres else 0
            score = base_scores.get(item_id, 0) + match_score
            scored_items.append((score, item_id))

        scored_items.sort(reverse=True)
        recommendations[user_id] = [item_id for _, item_id in scored_items[:k]]

    return recommendations


def run_inference(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    artifact_dir = ensure_directory(Path(args.artifact_dir))
    paths = {
        "users": data_dir / "users.csv",
        "items": data_dir / "items.csv",
        "interactions": data_dir / "interactions.csv",
    }

    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError("Data files missing; run the training mode first")

    users_df = pd.read_csv(paths["users"])
    items_df = pd.read_csv(paths["items"])
    interactions_df = pd.read_csv(paths["interactions"])

    train_df, test_df = split_interactions(interactions_df, args.test_ratio)
    base_scores = compute_base_scores(items_df, train_df)
    user_histories = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    eligible_users = test_df["user_id"].unique().tolist()
    if not eligible_users:
        raise ValueError("No users have test interactions; try generating more data")

    sample_users = min(args.sample_users, len(eligible_users))
    selected_users = random.sample(eligible_users, sample_users)

    latencies_ms = []
    recommendations = {}
    for user_id in selected_users:
        start = time.perf_counter()
        user_recs = build_recommendations(
            users_df[users_df["user_id"] == user_id],
            items_df,
            base_scores,
            user_histories,
            args.k,
        )
        duration = (time.perf_counter() - start) * 1000
        latencies_ms.append(duration)
        recommendations[user_id] = user_recs[user_id]

    rec_path = artifact_dir / "recommendations.json"
    rec_path.write_text(json.dumps(recommendations, indent=2))

    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0
    quantiles_n = min(len(latencies_ms), 20)
    if quantiles_n >= 2:
        p95_latency = statistics.quantiles(latencies_ms, n=quantiles_n)[-1]
    else:
        p95_latency = max(latencies_ms) if latencies_ms else 0

    latency_summary = {
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "sample_users": sample_users,
        "k": args.k,
        "test_ratio": args.test_ratio,
    }
    latency_path = artifact_dir / "inference_latency.json"
    latency_path.write_text(json.dumps(latency_summary, indent=2))

    logger.info("Inference completed (avg=%.2fms, p95=%.2fms) and recommendations saved to %s", latency_summary["avg_latency_ms"], latency_summary["p95_latency_ms"], rec_path)


def run_metrics(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    artifact_dir = ensure_directory(Path(args.artifact_dir))
    rec_path = artifact_dir / "recommendations.json"

    if not rec_path.exists():
        raise FileNotFoundError("No recommendations found; run the inference mode first")

    interactions_df = pd.read_csv(data_dir / "interactions.csv")
    items_df = pd.read_csv(data_dir / "items.csv")

    _, test_df = split_interactions(interactions_df, args.test_ratio)

    with open(rec_path) as f:
        recommendations = json.load(f)

    results = {
        "k_values": args.k_values,
        "recall": {},
        "coverage": calculate_coverage(recommendations, len(items_df)),
        "diversity": calculate_diversity(recommendations, items_df),
        "sample_size": len(recommendations),
        "test_rows": len(test_df),
    }

    for k in args.k_values:
        results["recall"][f"recall@{k}"] = round(calculate_recall_at_k(recommendations, test_df, k), 4)

    metrics_path = artifact_dir / "evaluation_results.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    logger.info("Metrics saved to %s", metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Local VertexRec pipeline runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    training = subparsers.add_parser("training", help="Generate data and train models locally")
    training.add_argument("--data-dir", type=str, default="data/local", help="Directory for synthetic datasets")
    training.add_argument("--artifact-dir", type=str, default="artifacts/local", help="Directory for pipeline artifacts")
    training.add_argument("--num-users", type=int, default=1000, help="Number of synthetic users")
    training.add_argument("--num-items", type=int, default=2000, help="Number of synthetic items")
    training.add_argument("--num-interactions", type=int, default=25000, help="Number of synthetic interactions")
    training.add_argument("--epochs", type=int, default=5, help="Training epochs for TF Recommenders")
    training.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension for TF models")
    training.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for TF models")
    training.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    training.add_argument("--n-estimators", type=int, default=50, help="Boosting rounds for XGBoost")
    training.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth for XGBoost")
    training.add_argument("--xgb-learning-rate", type=float, default=0.1, help="Learning rate for XGBoost")
    training.add_argument("--xgb-objective", type=str, default="reg:squarederror", help="XGBoost objective")
    training.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data")

    inference = subparsers.add_parser("inference", help="Run local inference to generate recommendations")
    inference.add_argument("--data-dir", type=str, default="data/local", help="Directory that holds synthetic datasets")
    inference.add_argument("--artifact-dir", type=str, default="artifacts/local", help="Artifacts directory where recommendations are stored")
    inference.add_argument("--k", type=int, default=10, help="Number of items to recommend per user")
    inference.add_argument("--sample-users", type=int, default=100, help="Number of users to score")
    inference.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of interactions reserved for evaluation")

    metrics = subparsers.add_parser("metrics", help="Evaluate recommendations using local data")
    metrics.add_argument("--data-dir", type=str, default="data/local", help="Directory that holds synthetic datasets")
    metrics.add_argument("--artifact-dir", type=str, default="artifacts/local", help="Artifacts directory with recommendations")
    metrics.add_argument("--test-ratio", type=float, default=0.2, help="Percentage of interactions for testing")
    metrics.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20], help="K values for evaluation")

    args = parser.parse_args()

    if args.mode == "training":
        train_models(args)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "metrics":
        run_metrics(args)
    else:
        parser.error("Unknown mode: %s" % args.mode)


if __name__ == "__main__":
    main()
