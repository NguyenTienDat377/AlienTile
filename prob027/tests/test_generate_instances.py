from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from generate_instances import (  # noqa: E402
    allocate_total,
    assign_size_tiers,
    generate_dataset,
    structural_score,
    target_from_clicks,
    uniform_item_order,
    witness_item_order,
)
from validate_dataset import validate_dataset  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def test_target_matches_running_example(self) -> None:
        clicks = [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ]
        expected = [
            [1, 1, 1, 0],
            [1, 0, 0, 2],
            [0, 2, 2, 2],
            [1, 0, 0, 2],
        ]
        self.assertEqual(target_from_clicks(clicks, 3), expected)

    def test_generation_is_reproducible_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            kwargs = {
                "sizes": [3, 4],
                "colours": [2, 3],
                "densities": [0.2, 0.5],
                "families": ["witness", "uniform"],
                "witness_count": 16,
                "uniform_count": 8,
                "master_seed": 270027,
            }
            first_summary = generate_dataset(first, **kwargs)
            second_summary = generate_dataset(second, **kwargs)
            self.assertEqual(first_summary, second_summary)

            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second) for path in second.rglob("*") if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative_path in first_files:
                self.assertEqual(
                    (first / relative_path).read_bytes(),
                    (second / relative_path).read_bytes(),
                )

            validation_summary, errors = validate_dataset(first)
            self.assertEqual(errors, [])
            self.assertEqual(validation_summary["instances"], 24)
            self.assertEqual(validation_summary["unique_targets"], 24)

    def test_witness_is_only_an_upper_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "dataset"
            generate_dataset(
                dataset_dir,
                sizes=[3, 4, 5],
                colours=[2],
                densities=[0.3],
                families=["witness"],
                witness_count=3,
                uniform_count=0,
                master_seed=7,
            )
            instance_path = next(dataset_dir.glob("*/*/*/*.json"))
            instance = json.loads(instance_path.read_text(encoding="utf-8"))
            self.assertIn("known_upper_bound", instance)
            self.assertNotIn("known_optimum", instance)
            self.assertEqual(
                instance["known_upper_bound"],
                sum(sum(row) for row in instance["witness_clicks"]),
            )

    def test_default_tiers_are_balanced_and_score_ordered(self) -> None:
        tiers = assign_size_tiers([4, 5, 6, 8, 10, 12], [2, 3, 4])
        counts = {tier: list(tiers.values()).count(tier) for tier in set(tiers.values())}
        self.assertEqual(counts, {"easy": 6, "medium": 6, "hard": 6})

        scores = {
            tier: [structural_score(n, c) for (n, c), value in tiers.items() if value == tier]
            for tier in ("easy", "medium", "hard")
        }
        self.assertLessEqual(max(scores["easy"]), min(scores["medium"]))
        self.assertLessEqual(max(scores["medium"]), min(scores["hard"]))

    def test_requested_family_totals_are_balanced_by_tier(self) -> None:
        sizes = [4, 5, 6, 8, 10, 12]
        colours = [2, 3, 4]
        densities = [0.1, 0.3, 0.6]
        tiers = assign_size_tiers(sizes, colours)
        witness_counts = allocate_total(witness_item_order(tiers, densities), 60)
        uniform_counts = allocate_total(uniform_item_order(tiers, colours), 6)

        witness_by_tier = {tier: 0 for tier in ("easy", "medium", "hard")}
        uniform_by_tier = {tier: 0 for tier in ("easy", "medium", "hard")}
        for (n, c, _density), count in witness_counts.items():
            witness_by_tier[tiers[(n, c)]] += count
        for (n, c), count in uniform_counts.items():
            uniform_by_tier[tiers[(n, c)]] += count

        self.assertEqual(witness_by_tier, {"easy": 20, "medium": 20, "hard": 20})
        self.assertEqual(uniform_by_tier, {"easy": 2, "medium": 2, "hard": 2})


if __name__ == "__main__":
    unittest.main()
