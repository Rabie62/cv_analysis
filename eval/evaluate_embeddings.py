"""Multilingual Embedding Benchmark for CV-Job Matching.

Compares candidate embedding models on:
1. Cross-lingual semantic similarity (FR CV sentence vs EN JD requirement)
2. Semantic discrimination (Relevant pair similarity vs Irrelevant pair similarity)
3. Latency (ms per text embedding)
4. Parameter size and footprint
"""

import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rich.console import Console
from rich.table import Table

console = Console()

CANDIDATE_MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Lightweight, 118M params, very fast"
    },
    {
        "name": "multilingual-e5-small",
        "model_id": "intfloat/multilingual-e5-small",
        "description": "E5 family, 118M params, tuned for passage/query retrieval"
    },
    {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "description": "Flagship multilingual SOTA, 560M params, dense+sparse"
    }
]

# Benchmark text pairs representing cross-lingual and in-domain CV vs JD scenarios
EVALUATION_PAIRS = [
    # 1. High Relevance Cross-lingual (FR CV -> EN JD)
    {
        "type": "Cross-Lingual Relevant",
        "text1": "Développé et déployé des microservices REST avec FastAPI et PostgreSQL sur GCP.",
        "text2": "Experience building scalable RESTful microservices with Python and relational databases.",
        "expected": "HIGH"
    },
    # 2. High Relevance In-domain English
    {
        "type": "Monolingual Relevant",
        "text1": "Designed distributed Redis caching layer saving infrastructure costs and lowering latency.",
        "text2": "Strong experience with caching systems like Redis to optimize high-throughput APIs.",
        "expected": "HIGH"
    },
    # 3. High Relevance In-domain French
    {
        "type": "Monolingual Relevant (FR)",
        "text1": "Mise en place de pipelines de réentraînement hebdomadaires et orchestration avec Apache Airflow.",
        "text2": "Solide expertise MLOps avec conteneurisation et orchestration de flux Airflow.",
        "expected": "HIGH"
    },
    # 4. Irrelevant / Distractor Pair (Software Engineer vs Graphic Design / Marketing)
    {
        "type": "Distractor Irrelevant",
        "text1": "Architected distributed Redis caching layer and migrated monolithic backend to FastAPI.",
        "text2": "Création de maquettes graphiques sur Figma, retouche photo Photoshop et identité visuelle.",
        "expected": "LOW"
    },
    # 5. Near-domain subtle mismatch (Frontend UI vs Deep Learning Computer Vision)
    {
        "type": "Subtle Mismatch",
        "text1": "Built interactive React components with Tailwind CSS and Redux state management.",
        "text2": "Entraînement de réseaux de neurones profonds convolutifs pour la segmentation d'images médicales.",
        "expected": "LOW"
    }
]


def evaluate_model(model_info: dict, device: str = "cpu"):
    console.print(f"\n[bold blue]Loading model:[/bold blue] {model_info['name']} on {device}...")
    start_load = time.perf_counter()
    model = SentenceTransformer(model_info["model_id"], device=device)
    load_time = time.perf_counter() - start_load

    results = []
    latencies = []

    for pair in EVALUATION_PAIRS:
        # If model is E5, prepend query: / passage: convention for optimal representation
        t1 = pair["text1"]
        t2 = pair["text2"]
        if "e5" in model_info["model_id"].lower():
            t1 = f"passage: {t1}"
            t2 = f"query: {t2}"

        t_start = time.perf_counter()
        emb1 = model.encode(t1, convert_to_tensor=True)
        emb2 = model.encode(t2, convert_to_tensor=True)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000
        latencies.append(latency_ms)

        cos_sim = util.cos_sim(emb1, emb2).item()
        results.append({
            "type": pair["type"],
            "similarity": cos_sim,
            "expected": pair["expected"]
        })

    avg_latency = float(np.mean(latencies))
    
    # Separation score: average relevant similarity minus average irrelevant similarity
    relevant_sims = [r["similarity"] for r in results if r["expected"] == "HIGH"]
    irrelevant_sims = [r["similarity"] for r in results if r["expected"] == "LOW"]
    margin = float(np.mean(relevant_sims) - np.mean(irrelevant_sims))

    return {
        "name": model_info["name"],
        "load_time_s": load_time,
        "avg_latency_ms": avg_latency,
        "margin": margin,
        "relevant_sim_avg": float(np.mean(relevant_sims)),
        "irrelevant_sim_avg": float(np.mean(irrelevant_sims)),
        "details": results
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold green]Starting Embedding Model Evaluation Benchmark (Device: {device})[/bold green]")

    summary_table = Table(title="Embedding Model Benchmark Results")
    summary_table.add_column("Model", style="cyan", no_wrap=True)
    summary_table.add_column("Avg Latency (ms)", justify="right")
    summary_table.add_column("Relevant Sim", justify="right")
    summary_table.add_column("Irrelevant Sim", justify="right")
    summary_table.add_column("Discrimination Margin", justify="right", style="bold green")

    all_metrics = []
    for m in CANDIDATE_MODELS:
        try:
            metrics = evaluate_model(m, device=device)
            all_metrics.append(metrics)
            summary_table.add_row(
                metrics["name"],
                f"{metrics['avg_latency_ms']:.1f}",
                f"{metrics['relevant_sim_avg']:.3f}",
                f"{metrics['irrelevant_sim_avg']:.3f}",
                f"{metrics['margin']:.3f}"
            )
        except Exception as e:
            console.print(f"[red]Error evaluating {m['name']}: {e}[/red]")

    console.print(summary_table)


if __name__ == "__main__":
    main()
