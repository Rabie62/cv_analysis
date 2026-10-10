"""Interactive CLI Demo for the CV Analysis Engine.

Runs end-to-end analysis on sample English and French CVs against target Job Descriptions.
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import CVAnalysisPipeline
from eval.test_samples import TEST_SAMPLES, TEST_JDS

console = Console()


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]CV Analysis & ATS Matcher - End-to-End Demo[/bold cyan]\n"
        "[dim]Model: qwen2.5:3b (Ollama local GPU) | Embeddings: MiniLM-L12 (CPU)[/dim]",
        border_style="cyan"
    ))

    # Load English Software Engineer Pair
    cv_sample = TEST_SAMPLES[0]
    jd_sample = TEST_JDS[0]

    console.print(f"\n[bold green]Ingesting CV:[/bold green] {cv_sample['description']}")
    console.print(f"[bold green]Target Job:[/bold green] {jd_sample['title']}")

    pipeline = CVAnalysisPipeline(
        llm_model="qwen2.5:3b",
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    console.print("\n[yellow]Running extraction, embedding matching, and scoring...[/yellow]")
    result = pipeline.analyze_cv_and_job(
        cv_text=cv_sample["raw_cv"],
        job_text=jd_sample["raw_text"]
    )

    matching = result["matching_result"]
    sub_scores = matching["sub_scores"]

    # 1. Display Scores
    console.print(f"\n[bold magenta]Overall Match Score:[/bold magenta] [bold yellow]{matching['overall_score']} / 100[/bold yellow]")
    
    score_table = Table(title="Score Breakdown & Dimensions")
    score_table.add_column("Dimension", style="cyan")
    score_table.add_column("Sub-Score", justify="right", style="bold")
    score_table.add_column("Evaluation Method", style="dim")

    score_table.add_row("Skills Match", f"{sub_scores['skills_match']}/100", "ESCO Taxonomy Normalization + Set Coverage")
    score_table.add_row("Experience Relevance", f"{sub_scores['experience_relevance']}/100", "Dense Multilingual Embedding Cosine Sim")
    score_table.add_row("Seniority Fit", f"{sub_scores['seniority_fit']}/100", "Years of Experience vs Required")
    score_table.add_row("Education Fit", f"{sub_scores['education_fit']}/100", "Degree Level & Discipline Match")
    score_table.add_row("Impact & Achievements", f"{sub_scores['impact_and_achievements']}/100", "Quantified Metrics Detection (%, $, ROI)")
    score_table.add_row("ATS Readiness", f"{sub_scores['ats_readiness']}/100", "Layout & Length Inspection")
    console.print(score_table)

    # 2. Display Strengths
    console.print("\n[bold green]Evidence-Backed Strengths:[/bold green]")
    for s in matching["strengths"]:
        verified_tag = "[green][VERIFIED][/green]" if s["is_verified_in_cv"] else "[red][UNVERIFIED][/red]"
        console.print(f"  • {verified_tag} {s['statement']}")
        if s["cv_evidence_quote"]:
            console.print(f"    [dim]Quote: \"{s['cv_evidence_quote']}\"[/dim]")

    # 3. Display Gaps
    if matching["gaps"]:
        console.print("\n[bold red]Gaps & Missing Criteria:[/bold red]")
        for g in matching["gaps"]:
            console.print(f"  • {g['statement']}")
    else:
        console.print("\n[bold green]No critical skill or seniority gaps detected![/bold green]")

    # 4. Display Recommendations
    recs = matching["recommendations"]
    if recs["bullet_rewrites"]:
        console.print("\n[bold blue]Recommended Bullet Point Enhancements (XYZ Formula):[/bold blue]")
        for b in recs["bullet_rewrites"]:
            console.print(f"  [dim]Original:[/dim]  {b['original_bullet']}")
            console.print(f"  [bold green]Rewrite:[/bold green]   {b['suggested_rewrite']}")
            console.print(f"  [dim]Rationale:[/dim] {b['rationale']}\n")

    console.print(f"[dim]Hallucination rejection rate: {matching['hallucination_rate']}%[/dim]")
    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    run_demo()
