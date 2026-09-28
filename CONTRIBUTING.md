# Contributing

Create a virtual environment and install `python -m pip install -e '.[dev]'`.
Run `python -m pytest` and `nrsaa demo --output outputs/check` before submitting changes. Choose a new demo directory for each run. Install `.[neural]` to run the optional TensorFlow test.

Keep test fixtures synthetic. Do not commit patient recordings, subject mappings, trained weights, credentials, or local run artifacts. Changes to the feature or label order must update the schema and model compatibility checks together.

When reporting model performance, include the dataset version, patient grouping, split, class support, threshold, and metric definition. Keep new evaluations separate from the historical capstone results.
