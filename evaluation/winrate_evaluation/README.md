## Winrate Evaluation 
path: _evaluation/winrate_evaluation/win_rate.py
This script evaluates multi-turn dialogues using **Google Gemini API**.
It scores each turn based on four key criteria:

* **Sensibleness**
* **Specificity**
* **Interestingness**
* **Transition Naturalness**

The model compares dialogue responses A and B within dialogue turns, determining which response performs better, or if they are tied.

```bash
pip install google-generativeai google-api-core
```

---

### 1. Input Format

The input must be a `.json` file with the following structure:

```json
[
  {
    "conversation_id": "conv_001",
    "turns": [
      {
        "ground_truth_intent": "{intent}", # optional
        "predicted_intent": "{intent}", # optional
        "integrated_test_dial": "<formatted dialogue A vs B>"
      },
      ...
    ]
  },
  ...
]
```

Each `turn` must include:

* `integrated_test_dial`: A dialogue snippet comparing two responses (A vs B).

---

### 2. Output

After execution, the following files are saved:

#### Evaluation Results

```json
{
  "<conversation_key>": {
    "conversation_id": "conv_001",
    "evaluations": [
      {
        "turn_idx": 0,
        "ground_truth_intent": "...",
        "predicted_intent": "...",
        "evaluation_result": {
          "Sensibleness": {
            "reasoning": "...",
            "score": "A" | "B" | "TIE"
          },
          "Specificity": { ... },
          "Interestingness": { ... },
          "Transition Naturalness": { ... }
        }
      },
      ...
    ]
  },
  ...
}
```

#### Error Logs

* `error_ids_gemini.pkl`: Python pickle file containing any conversation turn IDs that failed evaluation.

---

### 3. Notes

* The script includes exponential backoff and jitter for error handling.
* Sleep intervals are used to avoid hitting rate limits.
* All API responses are printed for traceability.
* JSON extraction is resilient to malformed Gemini responses.

---
