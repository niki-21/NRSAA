# Data and reproducibility

## Input contract

Supply a directory of matching PhysioNet-style `.hea` and `.mat` files. Nested directories are searched recursively. MATLAB files must contain a numeric `val` matrix shaped `[leads, samples]` matching the header. Leads **I, II, III** must be named in the header; their physical ordering can differ.

The loader reads ADC gain, baseline, and `/mV` units, then converts selected channels to millivolts. Other units, non-finite samples, inconsistent dimensions, duplicate leads, and missing paired files cause explicit errors. It accepts both `#Dx:` and `# Dx:` metadata forms. Unknown demographic values remain missing and are imputed from training data only.

```text
data/raw/
  dataset-a/
    example.hea
    example.mat
  dataset-b/
    another.hea
    another.mat
```

Get recordings from the [PhysioNet 2020 dataset](https://physionet.org/content/challenge-2020/1.0.2/) under the applicable data terms. Do not commit raw data, patient mappings, derived features, or model artifacts. The data in local project folders is not automatically approved for redistribution.

## Patient groups

Create `data/groups.json`, mapping relative recording paths **without the extension** to patient groups:

```json
{
  "dataset-a/example": "dataset-a:subject-001",
  "dataset-b/another": "dataset-b:subject-017"
}
```

Use the same group for all records belonging to the same person, including across sources when that linkage is known. Do not infer patient identity from arbitrary recording filenames. Every retained recording must have a group entry when a mapping is supplied.

Without groups, the dataset is marked `record` and training refuses to proceed unless `--allow-record-split` is supplied. That option is for exploratory checks and does not establish patient-independent performance.

## Feature preparation

1. Select named leads I, II, III and calibrate to mV.
2. Resample to 500 Hz, truncate to ten seconds, denoise with sym5 detail soft thresholding, then pad short recordings with zeros.
3. Encode the nine report labels as a multi-hot vector. Ignore diagnoses outside this scope; count and skip recordings with no target label. PAC and SVPB are separate here, matching the report rather than official Challenge equivalence scoring.
4. Compute 53 features: two demographic values and 17 per lead, covering amplitude, spectrum, wavelet detail energy, and heuristic RR statistics.
5. Save compressed NumPy arrays with ordered feature/label names, record IDs, group IDs, and processing metadata. Loading uses `allow_pickle=False`.

With `--include-signals`, the dataset also stores a `[records, 5000, 3]` float32 tensor. This takes about 60 KB per recording before compression. Preparation holds the full tensor in memory; use small subsets or ample memory. The default feature-only mode is substantially smaller.

## Evaluation

The random-forest baseline uses an 80/20 group split and separate binary classifiers with balanced class weights. Median imputation and missingness indicators are learned from training data. An entirely absent training label produces zero positive probabilities rather than an indexing error.

The LSTM uses approximately 60/20/20 group splits. Per-lead normalization is fitted on training signals only. Early stopping monitors validation loss; the held-out test set is used after training. Class coverage is not guaranteed by group shuffling, so review per-label support before interpreting metrics.

All labels use threshold 0.5. Exact-match accuracy requires every label for a record to agree. Hamming loss measures the fraction of individual label decisions that disagree. Undefined precision/recall/F1 values use zero; average precision is null for labels with no test positives.

## Artifacts

Run outputs are local and ignored by Git. They can contain recording identifiers and sensitive derived data.

| Output | Purpose |
| --- | --- |
| `metrics.json` | Explicit metric names and run metadata |
| `split.json` | Record/group membership for auditing separation |
| `predictions.npz` | Held-out targets and ordered probabilities |
| `model.joblib` | Baseline pipeline including fitted imputation |
| `model.keras` | Optional trained LSTM |
| `normalization.npz` | LSTM scaling parameters and label order |
| `history.json` | LSTM training/validation loss history |

Only load trusted serialized models. Keep datasets, trained models, and their exact environment together; cross-version model compatibility is not guaranteed. `requirements-tested.txt` records the dependency versions used during local validation, while `pyproject.toml` declares supported installation ranges.
