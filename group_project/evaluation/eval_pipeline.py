"""Evaluate the RAG pipeline with the four standard RAGAS metrics.

Outputs:
    evaluation_predictions.json  Cached RAG answers and retrieved contexts
    ragas_matrix.csv             Per-question RAGAS metric matrix
    ragas_matrix.json            Same matrix in JSON format
    results.md                   Aggregate scores and per-question table
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALUATION_DIR.parent.parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
PREDICTIONS_PATH = EVALUATION_DIR / "evaluation_predictions.json"
MATRIX_CSV_PATH = EVALUATION_DIR / "ragas_matrix.csv"
MATRIX_JSON_PATH = EVALUATION_DIR / "ragas_matrix.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_environment() -> None:
    """Load API keys from the project .env file."""
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Run: pip install python-dotenv") from exc
    load_dotenv(PROJECT_DIR / ".env")


def load_golden_dataset(limit: int | None = None) -> list[dict]:
    """Load and validate question, reference answer, and reference context."""
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {GOLDEN_DATASET_PATH}")
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must contain a JSON array")
    required = {"question", "expected_answer", "expected_context"}
    for index, row in enumerate(data, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Dataset row {index} is missing: {sorted(missing)}")
    return data[:limit] if limit is not None else data


def _context_records(sources: list[dict]) -> list[dict]:
    """Keep the exact retrieved chunks and their scores for auditability."""
    records = []
    for source in sources:
        records.append({
            "content": str(source.get("content") or ""),
            "score": source.get("score"),
            "score_type": source.get("score_type"),
            "raw_scores": source.get("raw_scores") or {},
            "retrieval_source": source.get("retrieval_source", source.get("source")),
            "metadata": source.get("metadata") or {},
        })
    return records


def _generate_semantic_only(question: str, top_k: int = 5) -> dict:
    """Generate with dense cosine retrieval only, using the same Task 10 prompt/LLM."""
    from src.task5_semantic_search import semantic_search
    from src.task10_generation import (
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
        _create_llm_client,
        format_context,
        reorder_for_llm,
    )

    sources = semantic_search(question, top_k=top_k)
    for source in sources:
        source["retrieval_source"] = "semantic"
        source["source"] = "semantic"
        source.setdefault("score_type", "cosine")
    reordered = reorder_for_llm(sources)
    context = format_context(reordered)
    user_message = f"""<context>
{context}
</context>

<question>
{question}
</question>"""
    client, model, _provider = _create_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {
        "answer": response.choices[0].message.content or "",
        "sources": reordered,
        "retrieval_source": "semantic",
        "context_used": context,
    }


def generate_predictions(golden: list[dict], configs: list[str]) -> list[dict]:
    """Run Hybrid and/or Semantic-only pipelines and cache all A/B outputs."""
    from src.task10_generation import generate_with_citation

    generators = {
        "hybrid": generate_with_citation,
        "semantic": _generate_semantic_only,
    }

    predictions = []
    total = len(golden) * len(configs)
    position = 0
    for config in configs:
        for item in golden:
            position += 1
            question = item["question"]
            print(f"[{position}/{total}] [{config}] Generating: {question[:65]}")
            try:
                result = generators[config](question)
                sources = _context_records(result.get("sources") or [])
                prediction = {
                    "config": config,
                    "question": question,
                    "answer": str(result.get("answer") or ""),
                    "contexts": [s["content"] for s in sources if s["content"]],
                    "reference_answer": item["expected_answer"],
                    "reference_context": item["expected_context"],
                    "retrieval_source": result.get("retrieval_source"),
                    "context_used": result.get("context_used", ""),
                    "retrieved_chunks": sources,
                    "error": None,
                }
            except Exception as exc:
                prediction = {
                    "config": config,
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "reference_answer": item["expected_answer"],
                    "reference_context": item["expected_context"],
                    "retrieval_source": config,
                    "context_used": "",
                    "retrieved_chunks": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {prediction['error']}")
            predictions.append(prediction)
            PREDICTIONS_PATH.write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return predictions


def load_cached_predictions(golden: list[dict], configs: list[str]) -> list[dict]:
    """Load a complete prediction cache created by generate_predictions()."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction cache not found: {PREDICTIONS_PATH}. Remove --reuse-predictions."
        )
    predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    questions = {row["question"] for row in golden}
    selected = []
    for config in configs:
        config_rows = [
            row for row in predictions
            if row.get("config") == config and row.get("question") in questions
        ]
        selected.extend(config_rows[: len(golden)])
    predictions = selected
    expected_count = len(golden) * len(configs)
    if len(predictions) < expected_count:
        raise ValueError(
            f"Cache has {len(predictions)} rows but evaluation needs {expected_count}"
        )
    return predictions[:expected_count]


def _build_ragas_judges():
    """Build OpenAI judge and embedding clients for RAGAS 0.1.x."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS evaluation")
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError("Run: pip install langchain-openai") from exc

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
    return (
        ChatOpenAI(model=judge_model, temperature=0),
        OpenAIEmbeddings(model=embedding_model),
    )


def evaluate_with_ragas(predictions: list[dict]):
    """Calculate the per-question 4-column RAGAS score matrix."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            'Install evaluation dependencies: pip install "ragas==0.1.21" datasets langchain-openai'
        ) from exc

    valid = [row for row in predictions if row["answer"] and row["contexts"]]
    if not valid:
        raise RuntimeError("No successful predictions with retrieved contexts to evaluate")

    dataset = Dataset.from_dict({
        "question": [row["question"] for row in valid],
        "answer": [row["answer"] for row in valid],
        "contexts": [row["contexts"] for row in valid],
        "ground_truth": [row["reference_answer"] for row in valid],
    })
    judge_llm, judge_embeddings = _build_ragas_judges()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    dataframe = result.to_pandas()
    dataframe.insert(0, "config", [row["config"] for row in valid])
    counters: dict[str, int] = {}
    test_ids = []
    for row in valid:
        config = row["config"]
        counters[config] = counters.get(config, 0) + 1
        test_ids.append(counters[config])
    dataframe.insert(1, "test_id", test_ids)
    return dataframe


def _safe_mean(dataframe, column: str) -> float:
    values = dataframe[column].dropna()
    return float(values.mean()) if len(values) else 0.0


def save_results(dataframe, predictions: list[dict]) -> None:
    """Persist the RAGAS matrix in machine-readable and Markdown formats."""
    dataframe.to_csv(MATRIX_CSV_PATH, index=False, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = dataframe.to_dict(orient="records")
    MATRIX_JSON_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    summaries = {}
    for config in dataframe["config"].unique():
        config_frame = dataframe[dataframe["config"] == config]
        metric_means = {
            metric: _safe_mean(config_frame, metric) for metric in metric_columns
        }
        metric_means["overall"] = sum(metric_means.values()) / len(metric_columns)
        summaries[config] = metric_means
    failures = [row for row in predictions if row.get("error")]

    lines = [
        "# RAGAS Evaluation Results",
        "",
        f"- Evaluated rows: {len(dataframe)}",
        f"- Pipeline errors: {len(failures)}",
        f"- Judge model: `{os.getenv('RAGAS_JUDGE_MODEL', 'gpt-4o-mini')}`",
        f"- Embedding model: `{os.getenv('RAGAS_EMBEDDING_MODEL', 'text-embedding-3-small')}`",
        "",
        "## Aggregate Matrix",
        "",
        "| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config, means in summaries.items():
        lines.append(
            f"| {config} | {means['faithfulness']:.4f} | "
            f"{means['answer_relevancy']:.4f} | {means['context_recall']:.4f} | "
            f"{means['context_precision']:.4f} | {means['overall']:.4f} |"
        )
    lines.extend([
        "",
        "## Per-question Matrix",
        "",
        "| Config | ID | Question | Faithfulness | Answer Relevancy | Context Recall | Context Precision |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for _, row in dataframe.iterrows():
        question = str(row["question"]).replace("|", "\\|").replace("\n", " ")
        values = [row.get(metric) for metric in metric_columns]
        formatted = ["N/A" if value != value else f"{float(value):.4f}" for value in values]
        lines.append(
            f"| {row['config']} | {int(row['test_id'])} | {question} | "
            + " | ".join(formatted) + " |"
        )


    if failures:
        lines.extend(["", "## Pipeline Errors", ""])
        for row in failures:
            lines.append(f"- **{row['question']}** — `{row['error']}`")

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline using RAGAS")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate the first N questions")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["hybrid", "semantic"],
        default=["hybrid", "semantic"],
        help="Retrieval configurations to compare (default: hybrid semantic)",
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse evaluation_predictions.json instead of calling Task 10 again",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")

    load_environment()
    golden = load_golden_dataset(args.limit)
    print(f"Loaded {len(golden)} test cases; configs: {', '.join(args.configs)}")
    predictions = (
        load_cached_predictions(golden, args.configs)
        if args.reuse_predictions
        else generate_predictions(golden, args.configs)
    )
    matrix = evaluate_with_ragas(predictions)
    save_results(matrix, predictions)
    print(f"Saved CSV matrix: {MATRIX_CSV_PATH}")
    print(f"Saved JSON matrix: {MATRIX_JSON_PATH}")
    print(f"Saved report: {RESULTS_PATH}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
