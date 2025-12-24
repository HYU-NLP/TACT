import json
import openai
import os
import time
import logging
from tqdm import tqdm

# Configure your OpenAI API key via environment variable
openai.api_key = ""


# File paths
INPUT_FILE = ""
OUTPUT_FILE = ""
LOG_FILE = ""

# Model and generation settings
MODEL_NAME = "gpt-4o-mini-2024-07-18"
MAX_TOKENS = 100
TEMPERATURE = 0.0

# Prompt template (insert user input into {human_input})
ZS_PROMPT = (
    "### Instructions ###\n"
    "You are an agent that responds to a user's message.\n"
    "Generate an appropriate response to the simulated scenario.\n"
    "Keep your responses short. Answer with one sentence. \n\n"
    "### Conversation ###\n"
    "{human_input}\n\n"
    "### Output ###\n"
    "Response : "
)

# Configure logging
testing_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=testing_format)


def load_data(file_path: str):
    """
    Load a list of dialogue items from a JSON file.
    Each item must include an "input" key.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, file_path: str):
    """
    Save enriched dialogue items to a new JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved output to {file_path}")
    print(f"Saved output to {file_path}")


def generate_response(user_input: str) -> str:
    """
    Build the prompt from user_input, call the OpenAI API, and return the model response.
    """
    prompt = ZS_PROMPT.format(human_input=user_input)
    messages = [{"role": "system", "content": prompt}]

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=1
    )
    return response.choices[0].message.content.strip()


def main():
    # Ensure API key is available
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Load input data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    data = load_data(INPUT_FILE)

    # Generate responses with progress bar
    for item in tqdm(data, desc="Generating responses", unit="item"):
        user_input = item.get("input", "")
        try:
            gpt_output = generate_response(user_input)
        except Exception as e:
            logging.error(f"Error generating for input '{user_input[:30]}...': {e}. Retrying...")
            time.sleep(5)
            try:
                gpt_output = generate_response(user_input)
            except Exception as e2:
                logging.error(f"Retry failed for input '{user_input[:30]}...': {e2}")
                gpt_output = ""

        # Store replies under two keys
        item["gpt-4o-mini"] = gpt_output
        logging.info(f"""{len(gpt_output)} chars /// {gpt_output}""")

    # Save to a new file
    save_data(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()
