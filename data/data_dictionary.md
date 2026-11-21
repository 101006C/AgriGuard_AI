# Data dictionary

## Prediction history JSONL
Each line in `prediction_history.jsonl` is one prediction event.

| Field | Type | Description |
|---|---|---|
| id | string | Unique prediction record identifier |
| timestamp | ISO-8601 string | Time the prediction was created |
| filename | string | Uploaded image filename |
| crop | string | Crop label parsed from model class |
| condition | string | Predicted condition label |
| status | string | Healthy or possible disease detected |
| confidence | number | Model's top-class score, percentage |
| advice | array of strings | General, cautious next-step guidance |
| top_predictions | array | Up to three predicted classes and scores |

Important: prediction confidence is not a calibrated disease probability or a confirmed diagnosis.
