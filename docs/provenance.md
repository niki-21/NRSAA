# Implementation notes and attribution

NRSAA packages the capstone's signal-processing, recurrent-classification, and sensor-acquisition work into a runnable Python prototype.

## Design decisions

- Signals are selected by lead name and calibrated to millivolts using the header's gain and baseline.
- Resampling and fixed-duration windows provide consistent input shapes.
- Demographic missingness is preserved until training-only imputation.
- Feature and label orders are persisted so training and inference agree.
- Patient groups separate training and evaluation records. Record-level evaluation requires an explicit flag.
- The LSTM uses three temporal channels and nine independent sigmoid outputs for simultaneous labels, with separate training, validation and test groups.
- Single-channel serial capture remains separate from three-lead classification. Serial arrival times are not a calibrated acquisition clock.

## Model lineage

The recurrent model builds on the earlier capstone LSTM(32) → Dropout(0.1) → Dense(32) → Dense(16) architecture. Its original four-class softmax was adapted to nine sigmoid outputs, binary cross-entropy, and three-channel temporal input.

The random forest provides an inspectable feature-based classifier. The final report's SE-ResNet and feature-based neural network are distinct models; their reported metrics are documented as historical results and are not attributed to either included classifier.

## Third-party contributions

The AF prototype references [Samin47/Real-time-Atrial-Fibrilation-prediction-deep-learning](https://github.com/Samin47/Real-time-Atrial-Fibrilation-prediction-deep-learning) and includes an MIT notice naming ZaberAyon (2019). That notice is retained for the recurrent and serial-capture adaptations.

The Arduino example credits Casey Kuhns / SparkFun Electronics (2014) and retains its original beerware notice. The capstone team and advisors are credited in the main README.

NRSAA's Python code and documentation are distributed under the [MIT License](../LICENSE). Third-party components retain their respective notices. Datasets and trained weights are not included or licensed by this repository.
