import json
import re
import csv
import asyncio
import time
import ast
from typing import List, Dict, Any

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


# =====================================================
# Existing helper functions
# =====================================================
def load_data(path: str) -> Dict[str, Any]:
    """Load a UTF-8-encoded JSON file that may contain a UTF-8 BOM."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def reconstruct_dialogue(parsed_data: List[Dict[str, Any]]) -> str:
    """
    Convert a list of utterance dictionaries back into a formatted dialogue
    string for logging or debugging.
    """
    lines = []
    for utt in parsed_data:
        line = f"{utt['turn']} [{utt['speaker']}]"
        # Add intent tags for USER turns
        if utt["speaker"] == "USER" and utt["intents"]:
            intents_str = " | ".join(utt["intents"])
            line += f" [{intents_str}]"
        # Add transition marker
        if utt["transition"] == "ToD":
            line += " [Transition to ToD]"
        elif utt["transition"] == "Chitchat":
            line += " [Transition to Chitchat]"
        line += f" {utt['content']}"
        lines.append(line)
    return "\n".join(lines)


def extract_json_from_markdown(text: str) -> List[Any]:
    """
    Extract fenced code blocks (``` ... ```) and parse them as JSON or Python
    literals.  Returns raw text for blocks that cannot be parsed.
    """
    pattern = r"```(.*?)```"
    blocks = re.findall(pattern, text, flags=re.DOTALL)

    results: List[Any] = []
    for block in blocks:
        block = block.strip()

        # (1) Try standard JSON
        try:
            results.append(json.loads(block))
            continue
        except json.JSONDecodeError:
            pass

        # (2) Fall back to Python literal
        try:
            results.append(ast.literal_eval(block))
        except Exception:
            results.append(block)
    return results


def error_json_parser(json_data: List[str]) -> Dict[str, Any]:
    """
    A last-resort parser for malformed JSON strings produced by the LLM.
    It removes the typical 'json\\n' prefix and then loads the result.
    """
    raw_json_str = json_data[0].replace("json\n", "")
    return json.loads(raw_json_str)


# =====================================================
# LLM-related setup
# =====================================================
CRITERIA_TEMPLATE = """
##Task Description
This augmented dataset is named TACT.
The TACT (TOD-And-Chitchat Transition) dataset was augmented to train dialogue models that can naturally and proactively handle transitions between task-oriented dialogue (TOD) and chitchat, reflecting the fluid nature of real-world conversations. 
Unlike existing datasets such as MultiWOZ, FusedChat, or InterfereChat—which contain limited or artificial mode transitions—TACT introduces diverse and realistic transition patterns, including both user- and agent-initiated switches. 
It also emphasizes recovery from interruptions and allows user preferences expressed during chitchat to influence subsequent task responses. 
By combining broader intent coverage (e.g., via SLURP) and seamless integration of dialogue modes, TACT aims to build more context-aware and human-like conversational agents.

##Evaluation Criteria:
**You are an evaluator that grades dialogues according to the following criteria:**
"dialogue_naturalness": (
    "1: Transition Naturalness – Smooth and context-driven mode switching (chitchat ↔ task)",
    "2: Task-Oriented Accuracy – Correctly fulfilling user requests and incorporating stated preferences",
    "3: Chitchat Quality & Relevance – Engaging, on-topic small talk that aligns with conversation flow",
    "4: Coherence & Context Tracking – Maintaining logical flow, remembering past details and avoiding contradictions",
    "5: Interruption Handling & Recovery – Managing unexpected topic shifts and resuming tasks without losing context",
    "6: Politeness & Appropriateness – Aligning tone and style with user expectations and social norms",
    "7: Naturalness & Human-Likeness – Fluency, spontaneity, and idiomatic usage without repetitive or robotic phrasing",
    "8: Overall User Satisfaction – Balancing efficient task completion with pleasant, context-rich conversation"
)


**You must:**
1. Provide a short reasoning describing any errors or strengths you see.
2. Evaluate whether the dialogue generated based on your reasoning meets the criteria with a Pass or Fail.

## Evaluation Step
Below is a concise template you can use when evaluating a dialogue in the TACT dataset. It includes a short reasoning phase and a pass/fail verdict for each of the eight criteria.

**1. Transition Naturalness (Chitchat ↔ Task)**
- **Short Reasoning:**  
  *(Briefly describe any notable strengths or errors, e.g., “The dialogue smoothly pivoted between small talk about weather and handling a restaurant booking, maintaining context.”)*  
- **Verdict:** Pass / Fail

**2. Task-Oriented Accuracy**
- **Short Reasoning:**  
  *(Evaluate how well the dialogue addressed user requests, including preference handling, e.g., “The system correctly incorporated the user’s vegetarian preference when searching for a restaurant.”)*  
- **Verdict:** Pass / Fail

**3. Chitchat Quality & Relevance**
- **Short Reasoning:**  
  *(Assess the small talk segments, e.g., “The chitchat about recent movies was relevant and flowed well from the user’s question.”)*  
- **Verdict:** Pass / Fail

**4. Coherence & Context Tracking**
- **Short Reasoning:**  
  *(Check for logical flow, references to past details, e.g., “System accurately remembered the user’s name and prior mention of an upcoming trip.”)*  
- **Verdict:** Pass / Fail

**5. Interruption Handling & Recovery**
- **Short Reasoning:**  
  *(Look for how the system manages tangential questions or topic shifts, e.g., “After a side conversation about weather, the system returned to booking a flight seamlessly.”)*  
- **Verdict:** Pass / Fail

**6. Politeness & Appropriateness**
- **Short Reasoning:**  
  *(Evaluate tone, courtesy, and formality, e.g., “All prompts used polite language and aligned with user expectations.”)*  
- **Verdict:** Pass / Fail

**7. Naturalness & Human-Likeness**
- **Short Reasoning:**  
  *(Look at linguistic fluency, idiomatic usage, e.g., “Responses felt spontaneous and varied, avoiding robotic repetition.”)*  
- **Verdict:** Pass / Fail

**8. Overall User Satisfaction**
- **Short Reasoning:**  
  *(Holistic assessment: Was the conversation both effective and pleasant, e.g., “The user got the info they wanted quickly, and the small talk was enjoyable.”)*  
- **Verdict:** Pass / Fail

Return output in this strict JSON format (no extra text):
{{
  "dialogue_naturalness":{{
      "reasoning": "reasoning",
      "score": <Pass or Fail>"
  }}
}}

##examples
**input_1**
0 [USER] [find_train] I will be traveling to Cambridge next month, and I need to find a train running from Bishop's Stortford heading to Cambridge, can you help me do this please?  
1 [SYSTEM] There are 70 trains doing that trip, do you have a date and time you would like to go?  
2 [USER] [find_train] Yes, that would be Monday after 16:30.  
3 [USER] [chitchat]  By the way, I really enjoy train journeys. There's something so relaxing about watching the scenery go by. Do you have a favorite route?  
4 [SYSTEM] That's a lovely perspective! Train journeys can be quite scenic, especially in the countryside. I don't have personal favorites, but many people enjoy routes through the Lake District or along the coast.  
5 [USER] [chitchat] That sounds amazing! I love nature views. I hope the train ride to Cambridge has some nice sights too.  
6 [SYSTEM] [Transition to ToD] I can check if the route from Bishop's Stortford to Cambridge offers any scenic views. Now, back to your train request. The earliest departure after 16:30 on Monday is TR8925, which will be departing Bishop's Stortford going to Cambridge at 17:29. Would you like to book a ticket?  
7 [USER] [book_train] Great can I get tickets for 6 people and the reference number?  
8 [SYSTEM] Booking was successful, the total fee is 60.59 GBP payable at the station. Your reference number is PO9S5J0V. Can I help you with anything else?  
9 [USER] [find_attraction] Thank you, I also am looking for an attraction called Saint John's College.  
10 [SYSTEM] Saint John's College is in the centre of town, their address is Saint John's Street. Do you need any other info on them?

**Output_1**
{{
  "dialogue_naturalness": {{
    "reasoning": "The dialogue features a smooth transition from task-oriented dialogue (finding a train) to chitchat about scenic train journeys, and then naturally returns to the task with a polite segue. The chitchat felt spontaneous and was contextually relevant to the train topic. The system correctly handled the task, remembered user preferences, and maintained coherence throughout. Task completion and interruption recovery were seamless, and tone was consistently polite and natural.",
    "score": "Pass"
  }}
}}


**input_2**
0 [USER] [find_train] I will be traveling to Cambridge next month, and I need to find a train running from Bishop's Stortford heading to Cambridge, can you help me do this please?  
1 [SYSTEM] There are 70 trains doing that trip, do you have a date and time you would like to go?  
2 [USER] [find_train] Yes, that would be Monday after 16:30.  
3 [USER] [chitchat]  By the way, I really enjoy train journeys. There's something so relaxing about watching the scenery go by. Do you have a favorite route?  
4 [SYSTEM] That's a lovely perspective! Train journeys can be quite scenic, especially in the countryside. I don't have personal favorites.
5 [USER] [chitchat] I love nature views. I hope the train ride to Cambridge has some nice sights too.  
6 [SYSTEM] [Transition to ToD] I can check if the route from Bishop's Stortford to Cambridge offers any scenic views. Now, back to your train request. The earliest departure after 13:30 on Monday is TR5725, which will be departing Bishop's Stortford going to Cambridge at 14:26. Would you like to book a tickets for 6 people?  
7 [USER] [find_train] can I get tickets for 6 people and the reference number?  
8 [SYSTEM] Booking was successful, the total fee is 60.59 GBP payable at the station. Your reference number is PO9S5J0V. Can I help you with anything else?  
9 [USER] [find_attraction] Thank you, I also am looking for an attraction called Saint John's College.  
10 [SYSTEM] Saint John's College is in the centre of town, their address is Saint John's Street. Do you need any other info on them?

**Output_2**

{{
  "dialogue_naturalness": {{
    "reasoning": "System suggested a train at 14:26 despite the user specifying after 16:30, which breaks task accuracy and coherence. While chitchat integration and tone were strong, this mismatch negatively affects naturalness and user experience.",
    "score": "Fail"
  }}
}}


Return output in this strict JSON format (no extra text):
{{
  "dialogue_naturalness":{{
      "reasoning": "reasoning",
      "score": <Pass or Fail>"
  }}
}}
"""

USER_PROMPT = """
Evaluate this dialogue
{dialogue}
"""


def get_openai_llm(
    llm_name: str, organization: str, api_key: str, temperature: float
) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client with the desired configuration."""
    return ChatOpenAI(
        model_name=llm_name,
        organization=organization,
        api_key=api_key,
        temperature=temperature,
    )


def get_criteria_chain(llm: ChatOpenAI) -> LLMChain:
    """Compose the System + Human prompts into a LangChain `LLMChain`."""
    system_prompt = SystemMessagePromptTemplate.from_template(
        template=CRITERIA_TEMPLATE
    )
    human_prompt = HumanMessagePromptTemplate.from_template(template=USER_PROMPT)
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return LLMChain(llm=llm, prompt=prompt)

# -----------------------------------------------------
# Async LLM call helper
# -----------------------------------------------------
async def generate_evaluation_async(
    criteria_chain: LLMChain, dialogue: str
) -> str:
    """
    Call the chain asynchronously.
    If your chain lacks an `arun` method, wrap a sync call in
    `asyncio.to_thread()`.
    """
    return await criteria_chain.arun(dialogue=dialogue)

# =====================================================
# Async workflow
# =====================================================
async def process_item(
    key: str,
    score_dict: Dict[str, Any],
    criteria_chain: LLMChain,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Evaluate a single dialogue with retry + rate-limit handling.
    The result is stored in-place under `score_dict[key]["evaluation_result"]`.
    """
    MAX_RETRY = 3
    attempt = 0
    generated_data = score_dict[key]["generated_data"]

    async with semaphore:
        while attempt < MAX_RETRY:
            try:
                # ----- LLM call ------------------------------------------------
                score = await generate_evaluation_async(criteria_chain, generated_data)

                # ----- Debug print --------------------------------------------
                print(f"\n=== Dialogue ID: {key} ===")
                print("GPT raw response:")
                print(score)
                print("================================\n")

                # ----- Parse JSON ---------------------------------------------
                parsed_score = json.loads(score)
                score_dict[key]["evaluation_result"] = parsed_score
                break  # Success!

            except Exception as e:
                msg = str(e).lower()

                # Handle OpenAI rate limits / throttling
                if any(token in msg for token in ("too many", "rate limit", "429")):
                    print(f"[{key}] Rate-limit error → retrying in 1 s …")
                    await asyncio.sleep(1)
                    attempt += 1
                    continue

                # Handle occasional malformed JSON using fallback parser
                if "string indices must be integers" in msg:
                    print(f"[{key}] Falling back to `error_json_parser` …")
                    score_dict[key]["evaluation_result"] = error_json_parser(
                        extract_json_from_markdown(score)  # type: ignore[arg-type]
                    )
                    break

                # Generic retry
                attempt += 1
                if attempt == MAX_RETRY:
                    print(f"[{key}] Aborted after {MAX_RETRY} attempts. Error: {e}")
                else:
                    print(f"[{key}] Error: {e} → retrying …")

# -----------------------------------------------------
# Main async entry point
# -----------------------------------------------------
async def main() -> None:
    """High-level orchestration: load data, run evaluations, save outputs."""
    # 1) Load data ------------------------------------------------------------------
    train_data = load_data(
        "/data/thskadud/dialogue_evauation/dialogue_naturalness/multiwoz_dict_for_geval_20250331.json"
    )

    # 2) Build LLM chain ------------------------------------------------------------
    llm = get_openai_llm(
        llm_name="gpt-4o-mini-2024-07-18",
        organization="your_organization",
        api_key="your_api_key",
        temperature=0.0,
    )
    criteria_chain = get_criteria_chain(llm)

    # 3) Concurrency guard -----------------------------------------------------------
    semaphore = asyncio.Semaphore(10)  # Max 10 concurrent calls

    # 4) Launch evaluation tasks -----------------------------------------------------
    tasks = [
        asyncio.create_task(
            process_item(key, train_data, criteria_chain, semaphore)
        )
        for key in train_data
    ]
    await asyncio.gather(*tasks)

    # 5) Persist results -------------------------------------------------------------
    # 5-A) JSON
    with open("TACT_multiwoz_dialogue_naturalness_20250331.json", "w", encoding="utf-8-sig") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    # 5-B) CSV
    rows: List[Dict[str, Any]] = []
    for file_name, content in train_data.items():
        split = content["split"]
        generated_data = content["generated_data"]
        source_data = content["source_data"]
        eval_result = content.get("evaluation_result", {})

        for eval_name, eval_content in eval_result.items():
            rows.append(
                {
                    "file_name": file_name,
                    "split": split,
                    "generated_data": generated_data,
                    "source_data": source_data,
                    "evaluation_type": eval_name,
                    "reasoning": eval_content.get("reasoning", ""),
                    "score": eval_content.get("score", ""),
                }
            )

    with open("dialogue_evaluation.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# =====================================================
# Entry point
# =====================================================
if __name__ == "__main__":
    asyncio.run(main())