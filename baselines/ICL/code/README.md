# ICL Implementation

> This repository contains **ICL (In-Context Learning)** based evaluation for MultiWOZ-style dialogue datasets.

---

### ✅ ICL Baseline

The ICL baseline enables inference using GPT-based in-context learning, without fine-tuning.

#### Scripts

- **gpt_chitchat.py**: Run ICL-based inference on chitchat classification
- **gpt_fs_TACT.py**: Few-shot TACT-style ICL
- **gpt_zs_TACT.py**: Zero-shot TACT-style ICL

Make sure you have proper OpenAI API keys configured before running.

---

## 🔗 Dependencies

```bash
pip install openai
pip install vllm
```
