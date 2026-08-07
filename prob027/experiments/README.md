# Alien Tiles benchmark generation

The benchmark generator uses only the Python standard library. It creates two
families of instances:

- `witness`: a sampled click matrix is converted to a target, so feasibility is
  guaranteed. `known_upper_bound` is not an optimum label.
- `uniform`: target cells are sampled uniformly and feasibility remains
  `unclassified` until an exact solver runs.

Instances are organised into `easy`, `medium` and `hard` tiers. The generator
computes the structural score `N^2 * (c - 1)`, sorts all configured `(N, c)`
pairs by this score, and divides the list into three balanced groups. This is a
model-size classification, not a claim about measured solver runtime.

Generate the compact 66-instance thesis benchmark:

```bash
python3 experiments/generate_instances.py \
  --output-dir data/benchmark_v1 \
  --sizes 4 5 6 8 10 12 --colours 2 3 4 \
  --densities 0.10 0.30 0.60 \
  --witness-count 60 --uniform-count 6 \
  --master-seed 270027
```

The exact allocation is 20 witness and 2 uniform-target instances in each
size tier. Every witness stratum receives one instance first; the six remaining
witness instances are spread evenly across tiers and densities.

The output directory must not already exist. Validate a generated dataset with:

```bash
python3 experiments/validate_dataset.py data/benchmark_v1
```

Run the minimisation solver recursively over one tier with:

```bash
python3 models/sat_variant2.py --input-dir data/benchmark_v1/easy
```

The generator writes `manifest.csv`, `dataset_summary.json`, a dataset README,
and one JSON file per instance. Per-instance seeds and sampling use SHA-256, so
the output is independent of Python's `random.Random` implementation.

Run the generator tests with:

```bash
python3 -m unittest discover -s tests -v
```
