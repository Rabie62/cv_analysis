"""Dataset Builder & Stratified Train/Val/Test Splitter.

Produces:
1. data/annotated/dataset.json (Full dataset of annotated CV-JD pairs)
2. data/splits/train.json (70%)
3. data/splits/val.json (15%)
4. data/splits/test.json (15%)
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.synthetic_generator import SyntheticDatasetGenerator

console = Console()
DATA_DIR = PROJECT_ROOT / "data"


def build_and_split_dataset(total_pairs: int = 160, seed: int = 42):
    random.seed(seed)
    annotated_dir = DATA_DIR / "annotated"
    splits_dir = DATA_DIR / "splits"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Generating {total_pairs} annotated CV/JD pairs...[/bold cyan]")
    dataset = SyntheticDatasetGenerator.generate_benchmark_dataset(target_total_pairs=total_pairs)

    # Save complete annotated dataset
    full_path = annotated_dir / "dataset.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Saved full annotated dataset to:[/green] {full_path} ({len(dataset)} pairs)")

    # Stratified split by (language, match_tier)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in dataset:
        group_key = f"{item['language']}_{item['match_tier']}"
        groups.setdefault(group_key, []).append(item)

    train_set, val_set, test_set = [], [], []

    for group_key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    # Save splits
    for name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        split_path = splits_dir / f"{name}.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Saved {name} split to:[/green] {split_path} ({len(split_data)} pairs)")

    # Display dataset statistics
    stats_table = Table(title="Dataset Split Statistics")
    stats_table.add_column("Split", style="cyan")
    stats_table.add_column("Total Pairs", justify="right")
    stats_table.add_column("English Pairs", justify="right")
    stats_table.add_column("French Pairs", justify="right")
    stats_table.add_column("Tier 1 (Strong)", justify="right")
    stats_table.add_column("Tier 2 (Mod)", justify="right")
    stats_table.add_column("Tier 3 (Pivot)", justify="right")
    stats_table.add_column("Tier 4 (Mismatch)", justify="right")

    for name, s in [("Train (70%)", train_set), ("Val (15%)", val_set), ("Test (15%)", test_set)]:
        en_c = sum(1 for x in s if x["language"] == "en")
        fr_c = sum(1 for x in s if x["language"] == "fr")
        t1 = sum(1 for x in s if x["match_tier"] == "tier_1_strong")
        t2 = sum(1 for x in s if x["match_tier"] == "tier_2_moderate")
        t3 = sum(1 for x in s if x["match_tier"] == "tier_3_pivot")
        t4 = sum(1 for x in s if x["match_tier"] == "tier_4_mismatch")
        stats_table.add_row(name, str(len(s)), str(en_c), str(fr_c), str(t1), str(t2), str(t3), str(t4))

    console.print(stats_table)
    return dataset, train_set, val_set, test_set


if __name__ == "__main__":
    build_and_split_dataset()
