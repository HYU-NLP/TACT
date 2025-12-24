import json
from typing import List, Dict, Any
import re
import ast
import asyncio
from asyncio import Semaphore

# from langchain.llms import Ollama
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

CRITERIA_TEMPLATE = """
##Task Description
This augmented dataset is named TACT.
The TACT (TOD-And-Chitchat Transition) dataset was augmented to train dialogue models that can naturally and proactively handle transitions between task-oriented dialogue (TOD) and chitchat, reflecting the fluid nature of real-world conversations. 
Unlike existing datasets such as MultiWOZ, FusedChat, or InterfereChat—which contain limited or artificial mode transitions—TACT introduces diverse and realistic transition patterns, including both user- and agent-initiated switches. 
It also emphasizes recovery from interruptions and allows user preferences expressed during chitchat to influence subsequent task responses. 
By combining broader intent coverage (e.g., via SLURP) and seamless integration of dialogue modes, TACT aims to build more context-aware and human-like conversational agents.


##Evaluation Criteria
**You are an evaluator reviewing a dialogue between a user and a system. Each user utterance is annotated with an intent tag. Your task is to evaluate whether the assigned intent tag for each user utterance is appropriate and accurate.**
- Consider the meaning of each user utterance in context.
- Assess whether the annotated intent correctly reflects what the user is trying to do or ask.
- Refer to the list of candidate intents to determine correctness.
- Provide reasoning for each evaluation to support your decision.

##Notes
- An intent is considered accurate if it clearly aligns with the user's intent in that turn.
- Mark each evaluation as either pass (if the intent is correct) or fail (if the intent is incorrect or ambiguous).

**candidate_intent**
{candidate_intent}, chitchat

##input_output_example
**input_example**
8 [USER] [book_restaurant] I am interested in making a reservation for 6 people at 18:00 on Saturday at a moderately priced restaurant.
9 [SYSTEM] Sounds like you have a fun evening planned, let me make sure we have the right place for you all. Any preference on price range?
10 [USER] [book_restaurant] I would like a moderately priced restaurant please.
11 [SYSTEM] I have booked you successfully at Jinling Noodle Bar on Saturday at 18:00. Your reference number is 2MOF0BGV. Is there anything else I could help you with?
12 [USER] [book_train] I also want a train to go to Stansted airport.

**output_example**
[
  {{
    "turn": 8,
    "speaker": "USER",
    "intents": [
      "book_restaurant"
    ],
    "reasoning": "USER is asking SYSTEM to make a restaurant reservation.",
    "evaluation": "pass"
  }},
  {{
    "turn": 10,
    "speaker": "USER",
    "intents": [
      "book_restaurant"
    ],
    "reasoning": "USER is giving details about a restaurant reservation.",
    "evaluation": "pass"
  }},
  {{
    "turn": 12,
    "speaker": "USER",
    "intents": [
      "book_train"
    ],
    "reasoning": "It's hard to tell from the information that they also want a train to Stansted Airport that they want a reservation. It makes more sense to think of it as looking for train information.",
    "evaluation": "Fail"
  }}
]
"""

USER_PROMPT = """
Evaluate this dialogue
{dialogue}
"""

# ═════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═════════════════════════════════════════════════════════════════════════════
def get_openai_llm(
    llm_name: str,
    organization: str,
    api_key: str,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client with the desired parameters."""
    return ChatOpenAI(
        model_name=llm_name,
        organization=organization,
        api_key=api_key,
        temperature=temperature,
    )


def get_criteria_chain(llm: ChatOpenAI) -> LLMChain:
    """Compose a System + Human prompt into an `LLMChain`."""
    criteria_prompt = SystemMessagePromptTemplate.from_template(template=CRITERIA_TEMPLATE)
    human_prompt = HumanMessagePromptTemplate.from_template(template=USER_PROMPT)
    prompt = ChatPromptTemplate.from_messages([criteria_prompt, human_prompt])
    return LLMChain(llm=llm, prompt=prompt)


def load_data(path: str) -> Dict[str, Any]:
    """Load a JSON file that may include a UTF-8 BOM."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def reconstruct_dialogue(parsed_data: List[Dict[str, Any]]) -> str:
    """
    Convert a list of utterance dictionaries back into a readable multi-line
    dialogue string.
    """
    lines = []
    for utt in parsed_data:
        line = f"{utt['turn']} [{utt['speaker']}]"
        # Append intent tags for USER turns
        if utt["speaker"] == "USER" and utt["intents"]:
            intents_str = " | ".join(utt["intents"])
            line += f" [{intents_str}]"
        # Append transition tags
        if utt["transition"] == "ToD":
            line += " [Transition to ToD]"
        elif utt["transition"] == "Chitchat":
            line += " [Transition to Chitchat]"
        # Append the actual content
        line += f" {utt['content']}"
        lines.append(line)
    return "\n".join(lines)


async def generate_evaluation_async(
    criteria_chain: LLMChain, dialogue: str, candidate_intent: str
) -> str:
    """Call the LLM asynchronously to get an evaluation JSON string."""
    return await criteria_chain.arun(
        candidate_intent=candidate_intent,
        dialogue=dialogue,
    )


def extract_json_from_markdown(text: str) -> List[Any]:
    """
    Extract ``` fenced blocks and attempt to parse each as JSON or a Python
    literal. Unparseable blocks are returned as raw strings.
    """
    pattern = r"```(.*?)```"
    blocks = re.findall(pattern, text, flags=re.DOTALL)

    results: List[Any] = []
    for block in blocks:
        block = block.strip()
        try:  # First try JSON
            results.append(json.loads(block))
            continue
        except json.JSONDecodeError:
            pass
        try:  # Fallback to Python literal
            results.append(ast.literal_eval(block))
        except Exception:
            results.append(block)
    return results


def error_json_parser(json_data: List[str]) -> Dict[str, Any]:
    """
    Fallback parser for malformed JSON that starts with a `json\\n` prefix.
    """
    raw_json_str = json_data[0].replace("json\n", "")
    return json.loads(raw_json_str)


# ═════════════════════════════════════════════════════════════════════════════
# Configuration / constants
# ═════════════════════════════════════════════════════════════════════════════
train_data = load_data(
    "/data/thskadud/dialogue_evauation/dialogue_naturalness/"
    "slurp_dict_for_geval_with_eval_blocks.json"
)
candidate_intents = load_data(
    "/data/thskadud/dialogue_evauation/intent_accuracy/"
    "slurp_candidate_intent_mapping.json"
)

MODEL_NAME = "gpt-4o-mini-2024-07-18"
ORGANIZATION = "your_organization"
API_KEY = "your_api_key"
TEMPERATURE = 0.0
MAX_CONCURRENT_REQUESTS = 10  # Semaphore limit

# Build the LLM chain
llm = get_openai_llm(MODEL_NAME, ORGANIZATION, API_KEY, TEMPERATURE)
criteria_chain = get_criteria_chain(llm)

# `score_dict` will be updated in-place with LLM evaluations
score_dict = train_data


# ═════════════════════════════════════════════════════════════════════════════
# Async processing helpers
# ═════════════════════════════════════════════════════════════════════════════
async def process_block(
    item_key: str,
    block_idx: int,
    target: str,
    candidate_intent: str,
    sem: Semaphore,
) -> None:
    """
    Evaluate a single dialogue *block* with retry logic.
    Results are written back into `score_dict` in-place.
    """
    async with sem:
        MAX_RETRY = 3
        attempt = 0
        while attempt < MAX_RETRY:
            try:
                score_raw = await generate_evaluation_async(
                    criteria_chain, target, candidate_intent
                )
                score = json.loads(score_raw)

                print(f"\n=== Dialogue ID: {item_key} ===")
                print("GPT raw response:")
                print(score)
                print("================================\n")

                # Store evaluation back into original structure
                score_dict[item_key]["eval_blocks"][block_idx] = {
                    "block": target,
                    "llm_eval_score": score,
                }
                break  # success

            except Exception as e:
                print(f"[retry {attempt + 1}/{MAX_RETRY}] error: {e}")
                attempt += 1

                # Handle a specific malformed-JSON symptom
                if "string indices must be integers" in str(e):
                    print("↳ Falling back to `error_json_parser` …")
                    fixed = error_json_parser(
                        extract_json_from_markdown(score_raw)  # type: ignore[arg-type]
                    )
                    score_dict[item_key]["eval_blocks"][block_idx] = {
                        "block": target,
                        "llm_eval_score": fixed,
                    }
                    break

                if attempt == MAX_RETRY:
                    print(f"⇢ Giving up on item '{item_key}'.")
                else:
                    print("⇢ Retrying after a short back-off …")
                    await asyncio.sleep(random.uniform(0.5, 1.5))


# ═════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ═════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    """Top-level coroutine that coordinates all evaluations."""
    sem = Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []

    for item_key, item_val in score_dict.items():
        eval_blocks = item_val.get("eval_blocks")
        if not eval_blocks:
            continue

        # You could look up candidate intents per block; here we use a stub
        candidate_intent = (
            "Decide if the annotated intent is appropriate based on the dialogue."
        )

        for block_idx, block in enumerate(eval_blocks):
            if block is None:
                continue
            target = reconstruct_dialogue(block)
            tasks.append(
                asyncio.create_task(
                    process_block(item_key, block_idx, target, candidate_intent, sem)
                )
            )

    await asyncio.gather(*tasks)

    # ---- Persist results ----------------------------------------------------
    class NumpyEncoder(json.JSONEncoder):
        """JSON encoder that converts NumPy types into native Python types."""
        def default(self, obj):  # type: ignore[override]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open("intent_accuracy_samples_2.json", "w", encoding="utf-8-sig") as f:
        json.dump(score_dict, f, cls=NumpyEncoder, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())