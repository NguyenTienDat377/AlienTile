#!/usr/bin/env python3
"""Generate reproducible benchmark instances for CSPLib prob027.

The generator creates two complementary instance families:

* ``witness``: sample a click matrix X, then compute its target.  These
  instances are feasible by construction, but the witness total is only an
  upper bound; it is not necessarily the optimum.
* ``uniform``: sample every target cell uniformly from {0, ..., c-1}.  The
  feasibility status is intentionally left unknown at generation time.

Instances are assigned to balanced ``easy``, ``medium`` and ``hard`` tiers by
the structural score N^2(c-1).  Every instance has a seed derived from its full
experimental coordinates, so changing loop order does not change existing
instances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "2.0"
GENERATOR_VERSION = "2.1"
FAMILIES = ("witness", "uniform")
SIZE_TIERS = ("easy", "medium", "hard")


class StableHashRNG:
    """Small SHA-256 counter-mode RNG with version-independent output."""

    def __init__(self, seed: int):
        if not 0 <= seed < 2**64:
            raise ValueError("seed must fit in 64 bits")
        self._key = seed.to_bytes(8, "big")
        self._counter = 0

    def _next_u64(self) -> int:
        payload = b"AlienTiles-RNG-v1" + self._key + self._counter.to_bytes(8, "big")
        self._counter += 1
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def randbelow(self, upper: int) -> int:
        """Return an unbiased integer in range(upper)."""
        if upper <= 0:
            raise ValueError("upper must be positive")
        space = 2**64
        limit = space - (space % upper)
        while True:
            value = self._next_u64()
            if value < limit:
                return value % upper

    def randint(self, lower: int, upper: int) -> int:
        if upper < lower:
            raise ValueError("upper must not be smaller than lower")
        return lower + self.randbelow(upper - lower + 1)

    def sample_indices(self, population_size: int, count: int) -> list[int]:
        """Sample distinct indices with a deterministic partial shuffle."""
        if not 0 <= count <= population_size:
            raise ValueError("invalid sample size")
        pool = list(range(population_size))
        for index in range(count):
            selected = index + self.randbelow(population_size - index)
            pool[index], pool[selected] = pool[selected], pool[index]
        return pool[:count]


def validate_matrix(matrix: list[list[int]], n: int, c: int, label: str) -> None:
    """Raise ValueError unless matrix is n-by-n with entries in [0, c-1]."""
    if len(matrix) != n:
        raise ValueError(f"{label}: expected {n} rows, got {len(matrix)}")
    for row_index, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"{label}: row {row_index} has {len(row)} values; expected {n}"
            )
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label}: values must be integers")
            if not 0 <= value < c:
                raise ValueError(f"{label}: value {value} is outside [0, {c - 1}]")


def target_from_clicks(clicks: list[list[int]], c: int) -> list[list[int]]:
    """Return the Alien Tiles target produced by a click matrix."""
    if c < 2:
        raise ValueError("c must be at least 2")
    n = len(clicks)
    if n < 1:
        raise ValueError("click matrix must be non-empty")
    validate_matrix(clicks, n, c, "clicks")

    row_sums = [sum(row) for row in clicks]
    column_sums = [sum(clicks[i][j] for i in range(n)) for j in range(n)]
    return [
        [
            (row_sums[i] + column_sums[j] - clicks[i][j]) % c
            for j in range(n)
        ]
        for i in range(n)
    ]


def stable_seed(master_seed: int, *coordinates: object) -> int:
    """Derive a stable 64-bit seed from the experimental coordinates."""
    payload = json.dumps(
        [master_seed, *coordinates], ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def target_sha256(n: int, c: int, target: list[list[int]]) -> str:
    """Hash the mathematical instance, independent of descriptive metadata."""
    payload = json.dumps(
        {"N": n, "c": c, "target": target},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sample_witness(n: int, c: int, density: float, rng: StableHashRNG) -> list[list[int]]:
    """Sample a click matrix with a controlled fraction of nonzero cells."""
    if not 0.0 < density <= 1.0:
        raise ValueError("witness density must be in (0, 1]")
    nonzero_count = max(1, min(n * n, round(density * n * n)))
    positions = rng.sample_indices(n * n, nonzero_count)
    clicks = [[0 for _ in range(n)] for _ in range(n)]
    for position in positions:
        i, j = divmod(position, n)
        clicks[i][j] = rng.randint(1, c - 1)
    return clicks


def sample_uniform_target(n: int, c: int, rng: StableHashRNG) -> list[list[int]]:
    """Sample a target uniformly from {0, ..., c-1}^{n*n}."""
    return [[rng.randbelow(c) for _ in range(n)] for _ in range(n)]


def target_statistics(target: list[list[int]], c: int) -> dict[str, Any]:
    values = [value for row in target for value in row]
    histogram = Counter(values)
    nonzero = sum(value != 0 for value in values)
    return {
        "nonzero_cells": nonzero,
        "nonzero_fraction": nonzero / len(values),
        "histogram": {str(value): histogram.get(value, 0) for value in range(c)},
    }


def density_token(density: float) -> str:
    return f"d{round(density * 1000):03d}"


def structural_score(n: int, c: int) -> int:
    """Return the model-size proxy N^2(c-1)."""
    return n * n * (c - 1)


def assign_size_tiers(
    sizes: Iterable[int], colours: Iterable[int]
) -> dict[tuple[int, int], str]:
    """Split configured (N, c) pairs into three balanced score-ordered tiers."""
    pairs = sorted(
        ((n, c) for n in sizes for c in colours),
        key=lambda pair: (structural_score(*pair), pair[0], pair[1]),
    )
    if len(pairs) < 3:
        raise ValueError("at least three (N, c) configurations are required")
    tiers: dict[tuple[int, int], str] = {}
    for rank, pair in enumerate(pairs):
        tier_index = min(2, rank * 3 // len(pairs))
        tiers[pair] = SIZE_TIERS[tier_index]
    return tiers


def interleave_tier_buckets(
    buckets: dict[str, list[tuple[Any, ...]]]
) -> list[tuple[Any, ...]]:
    """Interleave easy, medium and hard items to spread any remainder."""
    result: list[tuple[Any, ...]] = []
    max_length = max((len(items) for items in buckets.values()), default=0)
    for index in range(max_length):
        for tier in SIZE_TIERS:
            if index < len(buckets[tier]):
                result.append(buckets[tier][index])
    return result


def witness_item_order(
    tier_map: dict[tuple[int, int], str], densities: list[float]
) -> list[tuple[int, int, float]]:
    """Order witness strata so extra samples are balanced by tier and density."""
    buckets: dict[str, list[tuple[Any, ...]]] = {tier: [] for tier in SIZE_TIERS}
    for tier_index, tier in enumerate(SIZE_TIERS):
        pairs = sorted(
            (pair for pair, value in tier_map.items() if value == tier),
            key=lambda pair: (structural_score(*pair), pair[0], pair[1]),
        )
        rotated_densities = densities[tier_index:] + densities[:tier_index]
        for pair_index in range(len(pairs)):
            for density in rotated_densities:
                n, c = pairs[pair_index]
                buckets[tier].append((n, c, density))
    return [tuple(item) for item in interleave_tier_buckets(buckets)]


def uniform_item_order(
    tier_map: dict[tuple[int, int], str], colours: list[int]
) -> list[tuple[int, int]]:
    """Order target strata to balance tiers and rotate colours."""
    buckets: dict[str, list[tuple[Any, ...]]] = {tier: [] for tier in SIZE_TIERS}
    for tier_index, tier in enumerate(SIZE_TIERS):
        pairs = [pair for pair, value in tier_map.items() if value == tier]
        colour_groups = {
            colour: sorted(
                (pair for pair in pairs if pair[1] == colour),
                key=lambda pair: (structural_score(*pair), pair[0]),
            )
            for colour in colours
        }
        rotated_colours = colours[tier_index:] + colours[:tier_index]
        max_group = max((len(group) for group in colour_groups.values()), default=0)
        for group_index in range(max_group):
            for colour in rotated_colours:
                group = colour_groups[colour]
                if group_index < len(group):
                    buckets[tier].append(group[group_index])
    return [tuple(item) for item in interleave_tier_buckets(buckets)]


def allocate_total(items: list[tuple[Any, ...]], total: int) -> dict[tuple[Any, ...], int]:
    """Allocate an exact total as evenly as possible over an ordered item list."""
    if total < 0:
        raise ValueError("requested instance count must be non-negative")
    if not items:
        if total:
            raise ValueError("cannot allocate instances without configurations")
        return {}
    base, remainder = divmod(total, len(items))
    return {
        item: base + (1 if index < remainder else 0)
        for index, item in enumerate(items)
    }


def build_instance(
    *,
    n: int,
    c: int,
    size_tier: str,
    score: int,
    family: str,
    density: float | None,
    replicate: int,
    master_seed: int,
    instance_seed: int,
    duplicate_attempt: int,
    target: list[list[int]],
    clicks: list[list[int]] | None,
) -> dict[str, Any]:
    family_token = "witness_" + density_token(density) if density is not None else family
    name = f"{size_tier}_n{n:02d}_c{c:02d}_{family_token}_r{replicate:03d}"
    digest = target_sha256(n, c, target)
    parameters: dict[str, Any] = {}
    if density is not None:
        parameters["requested_nonzero_click_fraction"] = density

    instance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": (
            "Feasible target derived from a sampled click witness"
            if family == "witness"
            else "Uniformly sampled target with feasibility unclassified"
        ),
        "N": n,
        "c": c,
        "target": target,
        "generation": {
            "generator": "experiments/generate_instances.py",
            "generator_version": GENERATOR_VERSION,
            "method": family,
            "size_tier": size_tier,
            "structural_score": score,
            "replicate": replicate,
            "master_seed": master_seed,
            "instance_seed": instance_seed,
            "duplicate_attempt": duplicate_attempt,
            "parameters": parameters,
            "target_sha256": digest,
        },
        "target_statistics": target_statistics(target, c),
        "guaranteed_solvable": family == "witness",
        "status_at_generation": (
            "feasible_by_construction" if family == "witness" else "unclassified"
        ),
    }

    if clicks is not None:
        witness_total = sum(sum(row) for row in clicks)
        instance.update(
            {
                "witness_clicks": clicks,
                "witness_total_clicks": witness_total,
                "known_upper_bound": witness_total,
            }
        )
    return instance


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def generate_dataset(
    output_dir: Path,
    *,
    sizes: Iterable[int],
    colours: Iterable[int],
    densities: Iterable[float],
    families: Iterable[str],
    witness_count: int,
    uniform_count: int,
    master_seed: int,
) -> dict[str, Any]:
    """Generate a complete benchmark directory and return its summary."""
    sizes = list(sizes)
    colours = list(colours)
    densities = list(densities)
    families = list(families)

    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new directory"
        )
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError("sizes must contain positive integers")
    if not colours or any(c < 2 for c in colours):
        raise ValueError("colours must contain integers >= 2")
    if not densities or any(not 0.0 < d <= 1.0 for d in densities):
        raise ValueError("densities must be in (0, 1]")
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError(f"families must be selected from {FAMILIES}")
    if witness_count < 0 or uniform_count < 0:
        raise ValueError("family counts must be non-negative")
    if "witness" in families and witness_count == 0:
        raise ValueError("witness_count must be positive when witness family is enabled")
    if "uniform" in families and uniform_count == 0:
        raise ValueError("uniform_count must be positive when uniform family is enabled")

    output_dir.mkdir(parents=True)
    tier_map = assign_size_tiers(sizes, colours)
    witness_counts = allocate_total(
        witness_item_order(tier_map, densities),
        witness_count if "witness" in families else 0,
    )
    uniform_counts = allocate_total(
        uniform_item_order(tier_map, colours),
        uniform_count if "uniform" in families else 0,
    )
    seen_targets: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    counts_by_tier: Counter[str] = Counter()
    counts_by_family: Counter[str] = Counter()

    for n in sizes:
        for c in colours:
            size_tier = tier_map[(n, c)]
            score = structural_score(n, c)
            family_parameters: list[tuple[str, float | None]] = []
            if "witness" in families:
                family_parameters.extend(("witness", density) for density in densities)
            if "uniform" in families:
                family_parameters.append(("uniform", None))

            for family, density in family_parameters:
                density_key = None if density is None else f"{density:.12g}"
                if family == "witness":
                    assert density is not None
                    replicate_count = witness_counts[(n, c, density)]
                else:
                    replicate_count = uniform_counts[(n, c)]
                for replicate in range(replicate_count):
                    for attempt in range(1000):
                        instance_seed = stable_seed(
                            master_seed,
                            n,
                            c,
                            family,
                            density_key,
                            replicate,
                            attempt,
                        )
                        rng = StableHashRNG(instance_seed)
                        if family == "witness":
                            assert density is not None
                            clicks = sample_witness(n, c, density, rng)
                            target = target_from_clicks(clicks, c)
                        else:
                            clicks = None
                            target = sample_uniform_target(n, c, rng)

                        digest = target_sha256(n, c, target)
                        if digest not in seen_targets:
                            break
                    else:
                        raise RuntimeError(
                            f"could not generate a unique target for N={n}, c={c}, "
                            f"family={family}; reduce the requested dataset size"
                        )

                    seen_targets.add(digest)
                    instance = build_instance(
                        n=n,
                        c=c,
                        size_tier=size_tier,
                        score=score,
                        family=family,
                        density=density,
                        replicate=replicate,
                        master_seed=master_seed,
                        instance_seed=instance_seed,
                        duplicate_attempt=attempt,
                        target=target,
                        clicks=clicks,
                    )
                    relative_path = Path(size_tier) / family / f"n{n:02d}_c{c:02d}" / (
                        instance["name"] + ".json"
                    )
                    write_json(output_dir / relative_path, instance)

                    manifest_rows.append(
                        {
                            "name": instance["name"],
                            "path": relative_path.as_posix(),
                            "size_tier": size_tier,
                            "structural_score": score,
                            "family": family,
                            "N": n,
                            "c": c,
                            "click_density": "" if density is None else density,
                            "replicate": replicate,
                            "instance_seed": instance_seed,
                            "target_sha256": digest,
                            "guaranteed_solvable": family == "witness",
                            "witness_total_clicks": (
                                "" if clicks is None else instance["witness_total_clicks"]
                            ),
                            "known_upper_bound": (
                                "" if clicks is None else instance["known_upper_bound"]
                            ),
                        }
                    )
                    counts_by_tier[size_tier] += 1
                    counts_by_family[family] += 1

    fieldnames = list(manifest_rows[0].keys())
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "config": {
            "sizes": sizes,
            "colours": colours,
            "families": families,
            "witness_densities": densities,
            "requested_witness_instances": witness_count,
            "requested_uniform_instances": uniform_count,
            "master_seed": master_seed,
            "tier_policy": {
                "score": "N^2 * (c - 1)",
                "assignment": "balanced thirds after sorting (N,c) by score",
                "pairs": [
                    {
                        "N": n,
                        "c": c,
                        "score": structural_score(n, c),
                        "size_tier": tier_map[(n, c)],
                    }
                    for n, c in sorted(
                        tier_map,
                        key=lambda pair: (structural_score(*pair), pair[0], pair[1]),
                    )
                ],
            },
        },
        "counts": {
            "total": len(manifest_rows),
            "by_size_tier": dict(sorted(counts_by_tier.items())),
            "by_family": dict(sorted(counts_by_family.items())),
        },
        "labelling_policy": {
            "witness_total_is": "known_upper_bound_only",
            "known_optimum_is_written_only_after_exact_proof": True,
            "uniform_target_status_at_generation": "unclassified",
        },
    }
    write_json(output_dir / "dataset_summary.json", summary)

    readme = f"""# Alien Tiles generated benchmark

This directory was generated deterministically by
`experiments/generate_instances.py` with master seed `{master_seed}`.

- Total instances: {len(manifest_rows)}
- Easy instances: {counts_by_tier['easy']}
- Medium instances: {counts_by_tier['medium']}
- Hard instances: {counts_by_tier['hard']}
- Witness-derived instances: {counts_by_family['witness']}
- Uniform-target instances: {counts_by_family['uniform']}

The size tiers are balanced thirds ordered by the structural score N^2(c-1).
`manifest.csv` is the experiment index. A witness proves feasibility and
provides only an upper bound; it does not prove optimality.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible Alien Tiles benchmark suite"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 8, 10, 12])
    parser.add_argument("--colours", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument(
        "--families", choices=FAMILIES, nargs="+", default=list(FAMILIES)
    )
    parser.add_argument(
        "--densities", type=float, nargs="+", default=[0.10, 0.30, 0.60]
    )
    parser.add_argument("--witness-count", type=int, default=60)
    parser.add_argument("--uniform-count", type=int, default=6)
    parser.add_argument("--master-seed", type=int, default=270027)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_dataset(
        args.output_dir,
        sizes=args.sizes,
        colours=args.colours,
        densities=args.densities,
        families=args.families,
        witness_count=args.witness_count,
        uniform_count=args.uniform_count,
        master_seed=args.master_seed,
    )
    print(f"Generated {summary['counts']['total']} instances in {args.output_dir}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
