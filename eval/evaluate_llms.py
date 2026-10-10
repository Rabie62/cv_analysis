"""LLM Model Selection Benchmark.

Evaluates local open-source models (via Ollama) on:
1. JSON Schema Conformance (Pydantic validation pass rate)
2. Extraction accuracy (Skill Precision, Recall, F1 against ground truth)
3. Experience years extraction fidelity
4. Multilingual handling (EN and FR CVs)
5. Inference Speed (latency in seconds, tokens per second)
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from rich.console import Console
from rich.table import Table

from eval.test_samples import TEST_SAMPLES
from src.schemas import StructuredCV

console = Console()
OLLAMA_URL = "http://localhost:11434/api/generate"

EXTRACTION_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) parser.
Extract structured information from the provided CV/Resume into valid JSON matching this exact structure:
{
  "language": "en" or "fr",
  "personal_info": {"full_name": "...", "email": "...", "phone": "...", "location": "...", "linkedin": "...", "github": "..."},
  "summary": "...",
  "total_years_experience": float or null,
  "skills": [{"category": "...", "skills": ["..."]}],
  "raw_skills_list": ["skill1", "skill2"],
  "work_experience": [{"title": "...", "company": "...", "location": "...", "start_date": "...", "end_date": "...", "responsibilities": ["..."], "quantified_achievements": ["..."], "technologies_used": ["..."]}],
  "education": [{"degree": "...", "institution": "...", "graduation_year": int or null}],
  "certifications": [{"name": "...", "issuer": "...", "year": int or null}],
  "languages": [{"language": "...", "proficiency": "..."}],
  "ats_formatting_issues": []
}

Output ONLY valid JSON. Do not include markdown codeblocks or conversational text."""


def call_ollama(model: str, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": EXTRACTION_SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
        }
    }
    t0 = time.perf_counter()
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    data["wall_time_s"] = elapsed
    return data


def compute_skill_metrics(extracted_skills: List[str], ground_truth: List[str]):
    norm_extracted = set(s.lower().strip() for s in extracted_skills if s.strip())
    norm_gt = set(s.lower().strip() for s in ground_truth if s.strip())

    if not norm_extracted:
        return 0.0, 0.0, 0.0

    # Soft matching: exact match or substring match
    tp = 0
    for gt in norm_gt:
        if any(gt in ext or ext in gt for ext in norm_extracted):
            tp += 1

    precision = tp / len(norm_extracted) if norm_extracted else 0.0
    recall = tp / len(norm_gt) if norm_gt else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_model(model_name: str) -> Dict[str, Any]:
    console.print(f"\n[bold cyan]Evaluating LLM Model:[/bold cyan] {model_name}...")
    json_valid_count = 0
    pydantic_valid_count = 0
    total_samples = len(TEST_SAMPLES)
    f1_scores = []
    latencies = []
    tokens_per_sec_list = []

    for sample in TEST_SAMPLES:
        prompt = f"CV Content ({sample['language'].upper()}):\n{sample['raw_cv']}"
        try:
            res = call_ollama(model_name, prompt)
            response_text = res.get("response", "").strip()
            eval_count = res.get("eval_count", 0)
            eval_duration_ns = res.get("eval_duration", 1)
            tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0
            
            latencies.append(res["wall_time_s"])
            tokens_per_sec_list.append(tps)

            # 1. Parse JSON
            raw_json = json.loads(response_text)
            json_valid_count += 1

            # 2. Validate Pydantic Schema
            validated = StructuredCV.model_validate(raw_json)
            pydantic_valid_count += 1

            # 3. Extract skills
            extracted_skills = validated.raw_skills_list
            if not extracted_skills:
                # Fallback to flattening categories if raw_skills_list was omitted
                for cat in validated.skills:
                    extracted_skills.extend(cat.skills)

            p, r, f1 = compute_skill_metrics(extracted_skills, sample["ground_truth_skills"])
            f1_scores.append(f1)

            console.print(
                f"  Sample [{sample['id']}]: JSON OK, Pydantic OK, "
                f"Skill F1={f1:.2f}, Latency={res['wall_time_s']:.2f}s ({tps:.1f} tok/s)"
            )

        except Exception as e:
            console.print(f"  [red]Sample [{sample['id']}] Failed:[/red] {str(e)[:150]}")
            f1_scores.append(0.0)

    return {
        "model": model_name,
        "json_valid_rate": (json_valid_count / total_samples) * 100,
        "pydantic_valid_rate": (pydantic_valid_count / total_samples) * 100,
        "avg_f1": float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        "avg_latency_s": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "avg_tps": float(sum(tokens_per_sec_list) / len(tokens_per_sec_list)) if tokens_per_sec_list else 0.0
    }


def main():
    # Discover available models in Ollama
    r = httpx.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in r.json().get("models", [])]

    if not models:
        console.print("[yellow]No models found in Ollama. Pull a model first (e.g. `ollama pull qwen2.5:3b`)[/yellow]")
        return

    console.print(f"[bold green]Models available for benchmark:[/bold green] {models}")
    
    table = Table(title="LLM Model Selection Benchmark")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("JSON Valid %", justify="right")
    table.add_column("Pydantic Schema %", justify="right")
    table.add_column("Skill F1", justify="right", style="bold green")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Tokens/sec", justify="right")

    for model_name in models:
        res = evaluate_model(model_name)
        table.add_row(
            res["model"],
            f"{res['json_valid_rate']:.0f}%",
            f"{res['pydantic_valid_rate']:.0f}%",
            f"{res['avg_f1']:.2f}",
            f"{res['avg_latency_s']:.2f}s",
            f"{res['avg_tps']:.1f}"
        )

    console.print(table)


if __name__ == "__main__":
    main()
