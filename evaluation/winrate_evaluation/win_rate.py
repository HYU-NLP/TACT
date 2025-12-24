import json
import ast
import time
from datetime import datetime
import os
import re
import random
from tqdm import tqdm
import argparse
import pickle

import google.generativeai as genai
from google.api_core import exceptions

from typing import Dict, Any

import logging

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class GeminiHandler:
    def __init__(self, api_key, model_name: str = 'gemini-2.5-pro'):
        """
        api_key: str or List[str]
        When a list is provided, the handler will rotate keys on repeated errors (e.g., 429).
        """
        # Normalize to list
        if isinstance(api_key, (list, tuple)):
            self.api_keys = list(api_key)
        else:
            self.api_keys = [api_key]
        if not self.api_keys:
            raise ValueError("At least one API key must be provided.")

        self.key_idx = 0
        self.model_name = model_name

        # Retry/backoff
        self.per_key_retries = 3  # per user's request: try up to 3 times per key before switching
        self.base_delay = 2
        self.max_delay = 120

        # Configure first key
        self._use_key(self.key_idx)

    def _use_key(self, idx: int):
        current_key = self.api_keys[idx]
        genai.configure(api_key=current_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"[API] Using API key #{idx+1}/{len(self.api_keys)}")

    def extract_json_from_response(self, text: str) -> Dict[str, Any]:
        json_match = re.search(r'``````', text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text
            
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'({[\s\S]*})', json_str)
                if json_match:
                    return json.loads(json_match.group(1))
            except:
                pass
            
            raise ValueError(f"JSON parsed failed: {json_str[:100]}...")
    
    def extract_retry_delay(self, error_message: str) -> int:
        """Extract recommended retry delay in seconds from an error message."""
        retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_message)
        if retry_match:
            return int(retry_match.group(1))
        return None
        
    def generate_with_retry(self, prompt: str, temperature: float = 0) -> Dict[str, Any]:
        """
        Per-key retry with rotation:
        - Try up to self.per_key_retries times per key (especially on 429 quota errors).
        - After exceeding per-key retries, rotate to the next key.
        - If all keys are exhausted, raise an exception so caller can mark failure.
        """
        total_keys = len(self.api_keys)
        keys_tried = 0

        while keys_tried < total_keys:
            attempt = 0
            jitter = 0
            while attempt < self.per_key_retries:
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    logger.info("-"*20 + " winrate " + "-"*20)
                    logger.info(response.text)
                    if "json" in prompt.lower() or "{" in prompt.lower():
                        raw = response.text.strip()

                        # Roughly slice out a JSON-like block (without heavy regex)
                        # The entire response may not be JSON; capture the outermost { ... } range
                        l = raw.find('{')
                        r = raw.rfind('}')
                        if l != -1 and r != -1 and r > l:
                            candidate = raw[l:r+1]
                        else:
                            # if no JSON block found, return original text
                            return {"raw_text": raw, "error": "No JSON-like block found"}

                        # 1) Try standard JSON parsing first
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as je:
                            logger.debug(f"json.loads failed: {je}")

                        # 2) If that fails, parse as Python literal (allowing trailing commas, single quotes, etc.)
                        #    Replace JSON keywords (null/true/false) → Python keywords (None/True/False)
                        py_like = (candidate
                                .replace(": null", ": None")
                                .replace(": true", ": True")
                                .replace(": false", ": False")
                                .replace(":null", ": None")
                                .replace(":true", ": True")
                                .replace(":false", ": False"))
                        try:
                            obj = ast.literal_eval(py_like)
                            # json.dumps(obj, ensure_ascii=False)
                            return obj
                        except Exception as le:
                            logger.error(f"literal_eval failed: {le}")
                            return {"raw_text": raw, "error": f"Parse failed: {le}"}
                    else:
                        return {"text": response.text}
                    # if "json" in prompt.lower() or "{" in prompt.lower():
                    #     return self.extract_json_from_response(response.text)
                    # else:
                    #     return {"text": response.text}

                except exceptions.ResourceExhausted as e:
                    # 429: quota exceeded — backoff within this key, then rotate after per_key_retries
                    attempt += 1
                    error_msg = str(e)
                    retry_delay = self.extract_retry_delay(error_msg)
                    if retry_delay:
                        delay = min(retry_delay, self.max_delay)
                    else:
                        delay = min(self.base_delay * (2 ** (attempt - 1)) + jitter, self.max_delay)
                        jitter = random.uniform(0, 1)
                    logger.warning(f"[Quota exceeded (429)] key #{self.key_idx+1}: 재시도 {attempt}/{self.per_key_retries}, {delay}초 후 재시도...")
                    time.sleep(delay)
                    continue

                except exceptions.ServiceUnavailable as e:
                    # 503: transient; count toward per-key attempts
                    attempt += 1
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(1, 3), self.max_delay)
                    logger.warning(f"[Server overloaded (503)] key #{self.key_idx+1}: retry {attempt}/{self.per_key_retries}, {delay}초 후 재시도...")
                    time.sleep(delay)
                    continue

                except json.JSONDecodeError:
                    attempt += 1
                    logger.error(f"[JSON parse error] key #{self.key_idx+1}: retry {attempt}/{self.per_key_retries}")
                    time.sleep(2)
                    continue

                except Exception as e:
                    attempt += 1
                    logger.error(f"[General error] key #{self.key_idx+1}: {str(e)} (retry {attempt}/{self.per_key_retries})")
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    time.sleep(delay)
                    continue

            # Rotate to next key after per-key retry budget exhausted
            keys_tried += 1
            if keys_tried >= total_keys:
                break
            self.key_idx = (self.key_idx + 1) % total_keys
            self._use_key(self.key_idx)
            logger.warning(f"[API] Switching to next API key ({self.key_idx+1}/{total_keys}) after repeated failures.")

        raise Exception("All API keys exhausted after per-key retries.")
    
    def evaluate_dialogue(self, output: str, input_text: str = "", reference: str = "N/A") -> Dict[str, Any]:
        """Build prompt and run with per-key retry/rotation."""
        custom_prompt = self._create_evaluation_prompt(output, input_text, reference)
        return self.generate_with_retry(custom_prompt)
            
    def _create_evaluation_prompt(self, output, input_text="", reference="N/A"):
        custom_prompt_template = """
**Instruction:**    
In this task, you will see some pieces of chat conversations between "USER" and "SYSTEM". Note that all conversations shown in this task are hypothetical, not real conversations from users. 
Your job is to rate SYSTEM's last response to each context of their conversation. 
Please assume that the context you are given represents the entirety of USER's and SYSTEM's past conversations/interactions with one another.
Score 1 if it meets the evaluation criteria and 0 if it doesn't.
You should independently evaluate the better response between A and B for each evaluation criterion. 
If you think it's similar quality, print it as a TIE.

**Criteria:**
"Does the response make sense?": (
    "\n1.Use your common sense here. Is the response completely reasonable in context?\n"
    "2.If anything seems off-confusing, illogical, out of context, or factually wrong-then rate it as Does not make sense.\n"
    "3.If in doubt, choose Does not make sense.\n" 
),
"Is the response specific?": (
    "\n1. You may be asked to assess whether the response is specific to a given context.\n"
    "2. For example:\n – if USER says "I love tennis" and SYSTEM responds "That's nice", then mark it as Not specific. That reply could be used in dozens of different contexts.\n – but if SYSTEM responds "Me too, I can't get enough of Roger Federer!" then mark it as Specific, since it relates closely to what you're talking about. \n"
    "3. If you're in doubt, or if the reply seems at all generic, rate it as Not specific.\n"
),
"Is the response interesting?": (
    "\n1. You may be asked to rate whether or not the response is interesting.\n"
    "2. Choose Interesting if the response would likely catch someone's attention or arouse curiosity; also use that rating for anything insightful, unexpected, or witty.\n"
    "3. If the response is monotonous and predictable, or if you're unsure, then pick Not interesting.\n"
), 
"Is the transition natural?": (
    "\n1. The transition refers to switching from a chitchat (free-form conversation) mode to a task-oriented dialogue (ToD) mode, or vice versa.\n"
    "2. A natural transition means it should not be abrupt or out-of-context; the switch should feel smoothly connected to the previous flow of conversation.\n"
    "3. If the conversation flow is suddenly broken or the transition seems random, it is Not natural.\n"
    "4. The transition takes place with respect to each other, and two speakers must faithfully answer each other.\n"
)


Return output in this strict JSON format (no extra text):
{{
    "Sensibleness":{{
        "reasoning": "<short reasoning>",
        "score": <system responds A or B or TIE>,
    }},
    "Specificity":{{
        "reasoning": "<short reasoning>",
        "score": <system responds A or B or TIE>,
    }},
    "Interestingness":{{
        "reasoning": "<short reasoning>",
        "score": <system responds A or B or TIE>,
    }},
    "Transition Naturalness":{{
        "reasoning": "<short reasoning>",
        "score": <system responds A or B or TIE>,
    }}
}}

=== Begin Evaluation ===

Dialogue to evaluate [ToD -> ChitChat -> ToD] (output):
---
{output}
---
User query (input [ToD]):
---
{input}
---
Reference:
---
{reference}
---

Now follow the instructions strictly, and produce your final JSON.
"""
        return custom_prompt_template.format(output=output, input=input_text, reference=reference)

def save_checkpoint(results: Dict[str, Any],
                    error_ids: list,
                    failed_conversations: Dict[str, Any],
                    suffix: str = "partial",
                    out_dir: str = ".") -> None:
    """Save intermediate results and error info with a timestamped filename inside out_dir."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(out_dir, f"win_rate_{suffix}_{ts}")
    # Results JSON
    with open(f"{base}_result.json", "w", encoding="utf-8-sig") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Error IDs (pickle)
    import pickle
    with open(f"{base}_error_ids.pkl", "wb") as f:
        pickle.dump(error_ids, f)
    # Failed conversations (JSON)
    with open(f"{base}_failed_conversations.json", "w", encoding="utf-8-sig") as f:
        json.dump(failed_conversations, f, indent=2, ensure_ascii=False)
    logger.info(f"[Checkpoint] Saved intermediate files with base name '{base}_*'")

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append one JSON object per line and fsync for durability."""
    rec = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8-sig") as f:
        f.write(rec)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", choices=["none", "convo", "both"], default="convo",
                        help="Show no bars, conversation-only bar, or conversation+turn bars")
    parser.add_argument("--turn-bar-threshold", type=int, default=20,
                        help="Only show turn-level tqdm when number of turns >= this value")
    parser.add_argument("--tqdm-mininterval", type=float, default=0.5,
                        help="Minimum refresh interval for tqdm (sec)")
    args = parser.parse_args()

    json_path = "./sample.json" # your input json path here

    api_key = [
        # "your_api_key"
    ]
    gemini_handler = GeminiHandler(api_key=api_key)

    # Prepare per-run output directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"winrate_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[Run] Output directory: {run_dir}")
    # Point 'runs/latest' to this run (best-effort)
    latest_link = os.path.join("runs", "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(run_dir), latest_link)
    except Exception as _e:
        logger.warning(f"Could not update latest symlink: {str(_e)}")

    with open(json_path, "r", encoding="utf-8-sig") as f:
        fused_data = json.load(f)
    
    results = {}
    error_ids = []  # (conversation_id, turn_idx)
    failed_conversations = {}  # {conversation_id: [failed_turn_idx, ...]}
    total_conversations = len(fused_data)
    logger.info(f"[Start] Total conversations: {total_conversations}")

    try:
        for conv_idx, (key, conv_data) in enumerate(
                tqdm(
                    fused_data.items(),
                    total=total_conversations,
                    desc="Conversations",
                    dynamic_ncols=True,
                    mininterval=args.tqdm_mininterval,
                    disable=(args.progress == "none")
                ),
                start=1
            ):
            conversation_id = conv_data.get("conversation_id", key)
            turns = conv_data.get("turns", [])
            num_turns = len(turns)

            results[key] = {
                "conversation_id": conversation_id,
                "evaluations": []
            }

            conversation_failed_turns = []
            _show_turn_bar = (args.progress == "both" and num_turns >= args.turn_bar_threshold)
            turn_iterable = tqdm(
                turns,
                total=num_turns,
                desc=f"Turns: {conversation_id}",
                dynamic_ncols=True,
                mininterval=args.tqdm_mininterval,
                leave=False,
                disable=not _show_turn_bar
            )
            for turn_idx, turn_data in enumerate(turn_iterable if _show_turn_bar else turns):

                integrated_test_dial = turn_data.get("integrated_test_dial", "")

                try:
                    eval_result_dict = gemini_handler.evaluate_dialogue(integrated_test_dial)

                    results[key]["evaluations"].append({
                        "turn_idx": turn_idx,
                        "ground_truth_intent": turn_data.get("ground_truth_intent"),
                        "predicted_intent": turn_data.get("predicted_intent"),
                        "evaluation_result": eval_result_dict
                    })

                    # Stream-save per turn (append-only, durable)
                    append_jsonl(os.path.join(run_dir, "win_rate_result.jsonl"), {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "key": key,
                        "conversation_id": conversation_id,
                        "turn_idx": turn_idx,
                        "ground_truth_intent": turn_data.get("ground_truth_intent"),
                        "predicted_intent": turn_data.get("predicted_intent"),
                        "evaluation_result": eval_result_dict
                    })

                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"\t[Error] conversation '{conversation_id}', turn {turn_idx}: {str(e)}")
                    error_ids.append((conversation_id, turn_idx))
                    conversation_failed_turns.append(turn_idx)

                    append_jsonl(os.path.join(run_dir, "error_ids_gemini.jsonl"), {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "conversation_id": conversation_id,
                        "turn_idx": turn_idx,
                        "error": str(e)
                    })

                    time.sleep(3)
                    # Continue to next turn

            if conversation_failed_turns:
                failed_conversations[conversation_id] = conversation_failed_turns
                logger.info(f"\t -> Conversation '{conversation_id}' had failed turns: {conversation_failed_turns}")

            # Autosave every 5 conversations to protect progress
            if conv_idx % 5 == 0:
                save_checkpoint(results, error_ids, failed_conversations, suffix="autosave", out_dir=run_dir)

    except KeyboardInterrupt:
        logger.warning("[Interrupted] KeyboardInterrupt received. Saving checkpoint before exit...")
        save_checkpoint(results, error_ids, failed_conversations, suffix="interrupt", out_dir=run_dir)
        raise

    except Exception as e:
        logger.error(f"[Fatal] Unexpected error: {str(e)}")
        save_checkpoint(results, error_ids, failed_conversations, suffix="fatal", out_dir=run_dir)
        # Re-raise after saving so callers see the error
        raise

    with open(os.path.join(run_dir, "win_rate_result.json"), "w", encoding="utf-8-sig") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to '{os.path.join(run_dir, 'win_rate_result.json')}'")
    logger.info(f"Streaming results also appended to '{os.path.join(run_dir, 'win_rate_result.jsonl')}'")

    with open(os.path.join(run_dir, "failed_conversations.json"), "w", encoding="utf-8-sig") as f:
        json.dump(failed_conversations, f, indent=2, ensure_ascii=False)
    logger.info(f"Failed conversations saved to '{os.path.join(run_dir, 'failed_conversations.json')}'")

    with open(os.path.join(run_dir, "error_ids_gemini.pkl"), "wb") as f:
        pickle.dump(error_ids, f)
    logger.info(f"Error IDs saved to '{os.path.join(run_dir, 'error_ids_gemini.pkl')}'")

if __name__ == "__main__":
    main()
