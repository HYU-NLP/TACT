## Dialogue Validation Pipeline 
path: _dialogue/validation/code/*.py

This script evaluates **TACT-style** (Task-Oriented Dialogue and Chitchat Transition) conversations for *naturalness / intent accuracy / transition sentence* using a LangChain-based LLM.
At the end of execution, it produces:

1. A JSON file containing the original data with evaluation results.
2. A CSV summary report.

---

### 1. Input Data Structure (`multiwoz_dict_for_geval_*.json`)

```json
{
  "<dialogue_id>": {
    "split": "train | dev | test",
    "generated_data": "0 [USER] ...",   // The dialogue to evaluate (one line per turn)
    "source_data":    "0 [USER] ...",   // The reference or source dialogue (optional)
    "...": "additional metadata allowed"
  },
  "...": { ... }
}
```

* **Key:** dialogue ID (e.g., filename or UUID)
* **`generated_data`:** must be a single string containing the dialogue in this format:

  ```
  0 [USER] [find_train] ...
  1 [SYSTEM] ...
  2 [USER] [chitchat] ...
  ```
* Other fields like `source_data` are preserved but not used in evaluation.

---

### 2. Output Format

#### 2.1 JSON (Per-dialogue results)

```json
{
  "<dialogue_id>": {
    "split": "train",
    "generated_data": "...",
    "source_data": "...",
    "evaluation_result": {
      "dialogue_naturalness": {
        "reasoning": "short reasoning text",
        "score": "Pass"   // or "Fail"
      }
    }
  }
}
```

#### 2.2 CSV (Tabular results)

| file\_name | split | generated\_data | source\_data | evaluation\_type      | reasoning | score |
| ---------- | ----- | --------------- | ------------ | --------------------- | --------- | ----- |
| 0041.json  | train | (string)        | (string)     | dialogue\_naturalness | ...       | Pass  |

---
### 3. Notes

* If the LLM returns invalid JSON, `error_json_parser()` attempts fallback parsing.
* Only **valid JSON or Python literal** outputs are accepted.
  If you modify the prompt, ensure it keeps the strict JSON format to avoid parsing errors.

