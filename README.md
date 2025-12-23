# Beyond Task-Oriented and Chitchat Dialogues: Proactive and Transition-Aware Conversational Agents

<div align=center>
  <img alt="Static Badge" src="https://img.shields.io/badge/TACT-1.0-blue">
  <img alt="GitHub License" src="https://img.shields.io/github/license/HYU-NLP/TACT">
  <img alt="Github Created At" src="https://img.shields.io/github/created-at/HYU-NLP/TACT">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HYU-NLP/TACT">
  <br>
</div>

#### Official Repository for "Beyond Task-Oriented and Chitchat Dialogues: Proactive and Transition-Aware Conversational Agents." [[Paper(ACL Anthology)]](https://aclanthology.org/2025.emnlp-main.672/) [[Paper(arXiv)]](https://arxiv.org/abs/2511.08835)
##### Yejin Yoon, Yuri Son, Namyoung So, Minseo Kim, Minsoo Cho, Chanhee Park, Seungshin Lee and Taeuk Kim. *Accepted to EMNLP2025 long paper*.
---
### 🆕 Hugging Face Datasets Hub

TACT is now available via Hugging Face Datasets! Check it out here: [HYU-NLP/TACT](https://huggingface.co/datasets/HYU-NLP/TACT)

You can easily load the dataset using HF `datasets` library: 

```python
from datasets import load_dataset

dataset = load_dataset("HYU-NLP/TACT", data_dir="TACT_multiwoz")
print(dataset["test"][0])
```
---
### Abstract

Conversational agents have traditionally been developed for either task-oriented dialogue (TOD) or open-ended chitchat, with limited progress in unifying the two. Yet, real-world conversations naturally involve fluid transitions between these modes. To address this gap, we introduce TACT (TOD-And-Chitchat Transition), a dataset designed for transition-aware dialogue modeling that incorporates structurally diverse and integrated mode flows. TACT supports both user- and agent-driven mode switches, enabling robust modeling of complex conversational dynamics.To evaluate an agent’s ability to initiate and recover from mode transitions, we propose two new metrics—Switch and Recovery.Models trained on TACT outperform baselines in both intent detection and mode transition handling. Moreover, applying Direct Preference Optimization (DPO) to TACT-trained models yields additionalgains, achieving 75.74% joint mode-intent accuracy and a 70.1% win rate against GPT-4o in human evaluation.These results demonstrate that pairing structurally diverse data with DPO enhances response quality and transition control, paving the way for more proactive and transition-aware conversational agents.

![Representative Figure](docs/main.png)

---
## Repository Structure
```
./
├── README.md
├── docs/
└── datasets
    ├── TACT_multiwoz
    │   ├── test.jsonl
    │   ├── train.jsonl
    │   └── validation.jsonl
    └── TACT_slurp
        ├── validation.jsonl
        ├── test.jsonl
        └── train.jsonl
```

## Citation
```{bibtex}
@inproceedings{yoon-etal-2025-beyond,
    title = "Beyond Task-Oriented and Chitchat Dialogues: Proactive and Transition-Aware Conversational Agents",
    author = "Yoon, Yejin  and
      Son, Yuri  and
      So, Namyoung  and
      Kim, Minseo  and
      Cho, Minsoo  and
      Park, Chanhee  and
      Lee, Seungshin  and
      Kim, Taeuk",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.672/",
    doi = "10.18653/v1/2025.emnlp-main.672",
    pages = "13291--13317",
    ISBN = "979-8-89176-332-6",
    abstract = "Conversational agents have traditionally been developed for either task-oriented dialogue (TOD) or open-ended chitchat, with limited progress in unifying the two. Yet, real-world conversations naturally involve fluid transitions between these modes. To address this gap, we introduce TACT (TOD-And-Chitchat Transition), a dataset designed for transition-aware dialogue modeling that incorporates structurally diverse and integrated mode flows. TACT supports both user- and agent-driven mode switches, enabling robust modeling of complex conversational dynamics.To evaluate an agent{'}s ability to initiate and recover from mode transitions, we propose two new metrics{---}Switch and Recovery.Models trained on TACT outperform baselines in both intent detection and mode transition handling. Moreover, applying Direct Preference Optimization (DPO) to TACT-trained models yields additionalgains, achieving 75.74{\%} joint mode-intent accuracy and a 70.1{\%} win rate against GPT-4o in human evaluation.These results demonstrate that pairing structurally diverse data with DPO enhances response quality and transition control, paving the way for more proactive and transition-aware conversational agents."
}
```
```
Yejin Yoon, Yuri Son, Namyoung So, Minseo Kim, Minsoo Cho, Chanhee Park, Seungshin Lee, and Taeuk Kim. 2025. Beyond Task-Oriented and Chitchat Dialogues: Proactive and Transition-Aware Conversational Agents. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13291–13317, Suzhou, China. Association for Computational Linguistics.
```

## Log

- 2025.12.23 **TACT** is now available in this repository. 

## License

TACT is derived from publicly available datasets, including SLURP and MultiWOZ.

- The text portion of SLURP is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
- MultiWOZ is released under the MIT License.

Accordingly, TACT is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
Users must provide appropriate attribution when using this dataset.
