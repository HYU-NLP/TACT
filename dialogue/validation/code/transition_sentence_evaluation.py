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
def load_data(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data

def reconstruct_dialogue(parsed_data):
    reconstructed_lines = []
    for utt in parsed_data:
        line = f"{utt['turn']} [{utt['speaker']}]"
        if utt["speaker"] == "USER" and utt["intents"]:
            intents_str = " | ".join(utt["intents"])
            line += f" [{intents_str}]"
        if utt["transition"] == "ToD":
            line += " [Transition to ToD]"
        elif utt["transition"] == "Chitchat":
            line += " [Transition to Chitchat]"
        line += f" {utt['content']}"
        reconstructed_lines.append(line)
    return "\n".join(reconstructed_lines)

def extract_json_from_markdown(text: str):
    pattern = r"```(.*?)```"
    blocks = re.findall(pattern, text, flags=re.DOTALL)

    results = []
    for block in blocks:
        block = block.strip()
        try:
            parsed = json.loads(block)
            results.append(parsed)
            continue
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(block)
            results.append(parsed)
        except Exception:
            results.append(block)

    return results

def error_json_parser(json_data):
    raw_json_str = json_data[0]
    raw_json_str = raw_json_str.replace("json\n", "")
    parsed_dict = json.loads(raw_json_str)
    return parsed_dict


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
"transition_sentence_evaluation":(
    "\n1. Coherence Check – Is the transition from Chitchat to ToD logically consistent?\n"
    "2. Avoid Artificial Intent Insertion – Verify if the transition attempts to insert intent artificially or \"cheat\" by revealing future user intent unnecessarily.\n"
    "3. User-Initiated Transitions – Task-oriented actions should align with previous user comments.\n"
    "4. If the transition sentence does not reveal a clear intent, it gets a low score.\n"
    "5. The Transition sentence must not over-assume user intent.\n"
    "6. Ensure a natural progression from Chitchat to ToD or ToD to Chitchat.\n"
)

**You must:**
1. Provide a short reasoning describing any errors or strengths you see.
2. Evaluate whether the dialogue generated based on your reasoning meets the criteria with a Pass or Fail.

##Evlauation Step
**Step 1: Coherence Check**  
- Assess if the transition between Chitchat and Task-Oriented Dialogue (ToD) is logically consistent.  
- Verify that the flow of the conversation makes sense and doesn’t feel jarring or forced.  

**Step 2: Avoid Artificial Intent Insertion**  
- Evaluate if the transition sentence reveals unnecessary future user intent.  
- The dialogue should not hint or explicitly mention actions that would only be relevant later in the conversation.

**Step 3: User-Initiated Transitions**  
- Check if the task-oriented actions align with user comments from the chitchat portion.  
- Ensure that the user has expressed some preference or idea in the chitchat that is reflected appropriately in the task-oriented follow-up.

**Step 4: Clarity of Intent**  
- Ensure the transition sentence does not leave the user’s intent unclear.  
- The dialogue should not have vague transitions that confuse the context.

**Step 5: Avoid Over-Assuming User Intent**  
- Evaluate if the transition over-assumes what the user wants to do next.  
- The system should not act based on inferred intentions without clear user input.

**Step 6: Natural Progression**  
- Ensure the transition from Chitchat to ToD (or vice versa) feels natural.  
- The conversation should not feel like a forced switch but should flow seamlessly.

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
 "transition_sentence_evaluation": {{
    "reasoning": "The transitions from task-oriented dialogue (finding a train) to chitchat and back are smooth, contextually appropriate, and logically consistent, enhancing conversational flow without artificial insertions. The dialogue naturally transitions without over-assuming user intent. The user's interests in scenery naturally influence the system’s offer to check scenic views, reflecting genuine intent revelation.",
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
  "transition_sentence_evaluation": {{
    "reasoning": "The system incorrectly transitions to task-oriented dialogue by providing train details for 13:30 and 14:26, despite the user explicitly requesting times after 16:30. This breaks logical consistency and coherence. The system over-assumes the user's intent by proposing an earlier booking time and automatically suggesting tickets for 6 people without prior confirmation from the user, not accurately reflecting the user's explicitly stated preferences.",
    "score": "Fail"
  }}
}}

Return output in this strict JSON format (no extra text):
{{
  "transition_sentence_evaluation":{{
      "reasoning": "reasoning",
      "score": <Pass or Fail>"
  }}
}}
"""

USER_PROMPT = """
Evaluate this dialogue
{dialogue}
"""



# ═════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═════════════════════════════════════════════════════════════════════════════
def get_openai_llm(
    llm_name: str,
    organization: str,
    api_key: str,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Instantiate a ChatOpenAI client."""
    return ChatOpenAI(
        model_name=llm_name,
        organization=organization,
        api_key=api_key,
        temperature=temperature,
    )


def get_criteria_chain(llm: ChatOpenAI) -> LLMChain:
    """Compose the system + user prompts into an `LLMChain`."""
    system_prompt = SystemMessagePromptTemplate.from_template(template=CRITERIA_TEMPLATE)
    human_prompt = HumanMessagePromptTemplate.from_template(template=USER_PROMPT)
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    return LLMChain(llm=llm, prompt=prompt)


async def generate_evaluation_async(chain: LLMChain, dialogue: str) -> str:
    """Asynchronously run the chain and return the raw JSON string."""
    return await chain.arun(dialogue=dialogue)


# ═════════════════════════════════════════════════════════════════════════════
# Asynchronous evaluation routine
# ═════════════════════════════════════════════════════════════════════════════
async def process_item(
    key: str,
    score_dict: Dict[str, Any],
    chain: LLMChain,
    sem: asyncio.Semaphore,
) -> None:
    """
    Evaluate a single dialogue with retry + rate-limit handling.
    Writes the parsed result back into `score_dict[key]["evaluation_result"]`.
    """
    MAX_RETRY = 3
    attempt = 0
    dialogue_text = score_dict[key]["generated_data"]

    async with sem:
        while attempt < MAX_RETRY:
            try:
                raw = await generate_evaluation_async(chain, dialogue_text)

                print(f"\n=== Dialogue ID: {key} ===")
                print("GPT raw response:")
                print(raw)
                print("================================\n")

                score_dict[key]["evaluation_result"] = json.loads(raw)
                break

            except Exception as e:
                msg = str(e).lower()

                # Rate-limit / 429 handling
                if any(t in msg for t in ("too many", "rate limit", "429")):
                    print(f"[{key}] Rate-limit detected → wait & retry")
                    await asyncio.sleep(1)
                    attempt += 1
                    continue

                # Malformed-JSON recovery path
                if "string indices must be integers" in msg:
                    print(f"[{key}] Attempting fallback JSON parser")
                    score_dict[key]["evaluation_result"] = error_json_parser(
                        extract_json_from_markdown(raw)  # type: ignore[arg-type]
                    )
                    break

                attempt += 1
                if attempt == MAX_RETRY:
                    print(f"[{key}] Aborted after {MAX_RETRY} attempts: {e}")
                else:
                    print(f"[{key}] Error: {e} → retrying …")


# ═════════════════════════════════════════════════════════════════════════════
# Main coroutine
# ═════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    """Top-level orchestration for dataset scoring."""
    # ── 1. Load data ────────────────────────────────────────────────────────
    score_dict = load_data(
        "/data/thskadud/dialogue_evauation/dialogue_naturalness/slurp_dict_for_geval.json"
    )

    # ── 2. Build the LLM chain ─────────────────────────────────────────────
    chain = get_criteria_chain(
        get_openai_llm(
            llm_name="gpt-4o-mini-2024-07-18",
            organization="your_organization",
            api_key="your_api_key",
            temperature=0.0,
        )
    )

    # ── 3. Create a semaphore to cap concurrent requests ───────────────────
    sem = asyncio.Semaphore(10)

    # ── 4. Launch evaluation tasks ─────────────────────────────────────────
    tasks = [
        asyncio.create_task(process_item(k, score_dict, chain, sem))
        for k in score_dict
    ]
    await asyncio.gather(*tasks)

    # ── 5. Persist results ─────────────────────────────────────────────────
    with open("TACT_SLURP_transition_sentence_eval.json", "w", encoding="utf-8-sig") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)

    # Save a flat CSV view for easy spreadsheet inspection
    rows = []
    for file_name, meta in score_dict.items():
        split = meta["split"]
        gen = meta["generated_data"]
        src = meta["source_data"]
        for name, res in meta.get("evaluation_result", {}).items():
            rows.append(
                {
                    "file_name": file_name,
                    "split": split,
                    "generated_data": gen,
                    "source_data": src,
                    "evaluation_type": name,
                    "reasoning": res.get("reasoning", ""),
                    "score": res.get("score", ""),
                }
            )

    with open("TACT_SLURP_transition_sentence_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())
