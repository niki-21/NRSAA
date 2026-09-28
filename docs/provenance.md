# Source provenance and cleanup

## Recovered materials

| Source | What it contributed | Disposition |
| --- | --- | --- |
| Senior design Git checkout | Three-lead conversion, wavelet denoising, label mapping, feature experiments | Consolidated into `src/ecg_monitor` with corrected units, input validation and a single CLI |
| Senior Design Code/Data notebooks | Earlier tabular heart-disease comparisons and peak-detection exploration | Reviewed for context; unfinished and unrelated notebooks omitted |
| COE 491 final report and poster | System architecture, nine-label scope, model progression and reported results | Summarized with report-page references in the research case study |
| AF prototype `lstm2.py` | LSTM(32), dropout, and two dense layers | Adapted into the optional recurrent experiment |
| AF prototype `reader.py` | Serial acquisition workflow | Replaced with bounded capture, correct byte timeout handling, event preservation and timestamps |
| AF prototype Arduino sketch | SparkFun AD8232 sensor example | Retained with its full original attribution and notice |

The original files and local checkpoints remain in their source folders. A snapshot of the prior publishable checkout was also retained outside this repository before restructuring. These local archives are not part of the public source tree.

## Deliberate code changes

The maintained package is a cleanup and engineering reconstruction of recovered components. It is not a byte-for-byte restoration of the final neural experiment.

- Replaced desktop-specific paths, import-time file operations, duplicated scripts and ad hoc CSV parsing with explicit command-line interfaces.
- Fixed output overwriting across datasets and selected channels by lead name instead of assuming their ordering.
- Introduced physical-unit calibration, real resampling, fixed windows, and reproducible feature/label schemas.
- Removed placeholder physiological features whose implementations were missing or did not measure their claimed quantity.
- Preserved missing demographics instead of inventing an age or sex; fit imputation only on the training partition.
- Replaced the incomplete XGBoost/random-forest script with an end-to-end multilabel baseline and persisted evaluation artifacts.
- Adapted the earlier four-class LSTM to three input channels and nine independent sigmoid outputs. Kept the 32-unit recurrent layer and 32/16-unit dense trunk; changed loss to binary cross-entropy and separated validation from test data.
- Retained the original sensor sketch as a hardware example, while treating single-channel serial capture separately from the three-lead model.

## Excluded material

The old challenge baseline, duplicate preprocessing files, scratch notebooks, checkpoints of uncertain training provenance, raw datasets, machine-specific metadata, and administrative documents are omitted from the maintained tree. The AF prototype's test script contained fixed printed accuracy and diagnosis strings; those are not computed results and are not used as evidence here.

The final report describes an SE-ResNet, weighted-loss experiments, and a feature-based dense network. Their final source and attributable checkpoints were not found in the inspected folders. Reported results are labeled historical until these artifacts and evaluation splits are recovered and reproduced.

## Attribution

The AF clone records [Samin47/Real-time-Atrial-Fibrilation-prediction-deep-learning](https://github.com/Samin47/Real-time-Atrial-Fibrilation-prediction-deep-learning) as its upstream and contains an MIT license naming ZaberAyon (2019). That notice is preserved under `third_party` for the recovered/adapted material.

The Arduino sketch credits Casey Kuhns / SparkFun Electronics (2014) and includes its original beerware notice. The complete header is retained. The capstone team and advisors are credited in the main README; no exclusive individual contribution claims are inferred.

A project-wide license has not been assigned. Retaining upstream notices does not select a license for the team's other code or documents. NRSAA starts with a fresh Git history containing only the curated source and documentation. Earlier repositories and their historical metadata are not imported.
