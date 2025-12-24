import json
import openai
import os
import re
import logging
import time
from tqdm import tqdm

# ——— Configure your API key ———
openai.api_key = ""

# ——— Logging setup ———
LOG_FILE = ".log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_conversations_json(file_path: str):
    """
    Reads a JSON file containing a list of dicts, each with an "input" field.
    Returns that list of dicts.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_conversation(conversation: dict) -> dict:
    conv_id = conversation.get("dialogue_id", "<no-id>")
    human_messages = conversation.get("input", "")

    zs_prompt = f"""
### Instructions ###
You are an agent that detects the intent of the user's message and responds accordingly.

1. Choose an intent that best fits the user's last utterance. Choose the intent from the intent list. Do not create a new intent.
   If the user's last utterance is not related to any of the intents, choose 'chitchat'.

   Intent List: ['add_contact',
        'adjust_music','ask_cooking', 'book_taxi',
        'book_ticket', 'change_light',
        'change_volume', 'check_alarm',
        'check_calendar', 'check_contact',
        'check_datetime', 'check_email',
        'check_food', 'check_lists',
        'check_news', 'check_social',
        'check_traffic', 'check_transport',
        'check_weather', 'convert_time',
        'createoradd_list', 'decrease_volume',
        'dim_light', 'find_recipe',
        'increase_light', 'increase_volume',
        'make_coffee', 'mute_volume',
        'order_food', 'play_audiobook',
        'play_game', 'play_music',
        'play_podcast', 'play_radio',
        'post_social', 'query_music',
        'recommend_events', 'recommend_locations',
        'recommend_movies', 'remove_alarm',
        'remove_event', 'remove_list',
        'send_email', 'set_alarm',
        'set_event', 'start_cleaner',
        'turnoff_light', 'turnoff_wemo',
        'turnon_light', 'turnon_wemo'
 ]

2. Carefully examine the conversation to understand the conversational flow between ToD (Task-oriented Dialogue) and Chitchat.  
   Decide whether a transition tag is needed: `[Transition to ToD]`, `[Transition to Chitchat]`, or `[None]`.

3. Generate a response that is appropriate for the user's last utterance.  Keep your responses short. Answer with one sentence.

### Conversation ###
{human_messages}

### Output ###
Intent : 
Transition tag : 
Response : 
"""

    logging.info(f"[{conv_id}] Prompting GPT.")
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "system", "content": zs_prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        gpt_out = resp.choices[0].message.content.strip()
        #logging.info(f"[{conv_id}] Received {len(gpt_out)} chars from GPT.")
        # Log the actual GPT output for debugging
        #logging.info(f"[{conv_id}] GPT output: {gpt_out}")
    except Exception as e:
        logging.error(f"[{conv_id}] Error on first try: {e}. Retrying in 5s...")
        time.sleep(5)
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "system", "content": zs_prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            gpt_out = resp.choices[0].message.content.strip()
            #logging.info(f"[{conv_id}] Retry succeeded, {len(gpt_out)} chars.")
            # Log the GPT output from retry
            #logging.info(f"[{conv_id}] GPT output (retry): {gpt_out}")
        except Exception as e2:
            #logging.error(f"[{conv_id}] Retry failed: {e2}. Giving up.")
            gpt_out = ""

    # parse out the three fields
    m = re.search(
        r"Intent\s*:\s*(.*?)\s*Transition tag\s*:\s*(.*?)\s*Response\s*:\s*(.*)",
        gpt_out, re.DOTALL
    )
    if m:
        intent_text, tag_text, reply_text = (g.strip() for g in m.groups())
    else:
        intent_text, tag_text, reply_text = "", "", gpt_out

    logging.info(f"""{len(gpt_out)} chars""")
    logging.info(f"intent : {intent_text} // tag : {tag_text} // response : {reply_text}")

    conversation["response"] = {
        "from":         "gpt4o-mini",
        "intent":       intent_text,
        "response_tag": tag_text,
        "response":     reply_text
    }
    return conversation


def main():
    input_path = ""
    if not os.path.isfile(input_path):
        logging.error(f"Input file not found: {input_path}")
        print(f"Input file not found: {input_path}")
        return

    output_path = ""

    logging.info(f"Loading data from {input_path}")
    conversations = load_conversations_json(input_path)

    updated = []
    for conv in tqdm(conversations, desc="Processing", unit="conv"):
        updated.append(process_conversation(conv))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    logging.info(f"Wrote {len(updated)} records to {output_path}")
    print(f"Wrote {len(updated)} records to {output_path}")


if __name__ == "__main__":
    main()
