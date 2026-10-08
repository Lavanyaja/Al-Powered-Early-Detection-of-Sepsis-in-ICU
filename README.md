# AI-Powered Early Detection of Sepsis in ICUs

Sepsis kills ~11 million people a year, and current scoring systems flag it
only after a patient has deteriorated. This project trains an LSTM on ICU
time-series data to predict sepsis risk hour by hour, and explains each
alert so clinicians can see which signals drove it.

## Features
- 2-layer LSTM (64→32) with per-hour risk output, trained on PhysioNet 2019 (40,336 ICU patients)
- 67 engineered features: delta trends, 6-hour rolling mean/std, Shock Index, BUN/Creatinine ratio
- Occlusion-based explainability: top contributing features, severity tiers (mild/moderate/severe), per-hour risk timeline
- Flask dashboard (127.0.0.1:8000) with patient list, risk timelines, and alerts
- Fully local training and inference (tested on Apple M1, 8 GB RAM)

## Results
| Metric | Value |
|---|---|
| AUROC | 0.91 |

## Tech Stack
Python 3.11 · TensorFlow/Keras · scikit-learn · NumPy · Pandas · PyArrow · Flask · Plotly

## API
`/api/summary` · `/api/patients` · `/api/patient/<pid>` · `/api/patient/<pid>/timeline` · `/api/notify/<pid>`

## Future Work
Federated learning, wearable/IoT integration, fairness constraints,
LLM-generated alert summaries, sepsis subtype classification, clinical trials.


> Research/academic project. Not a certified medical device.
