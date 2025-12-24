# TACT Human Evaluation Interface

A simple web application for interacting with OpenAI's model, with support for system prompts and dialog history.

# Instruction
- requirement install
- set `.env`: openai api key, model path, ...
- run python app.py

## Features

- Send messages to GPT-4o-mini
- Set system prompts to guide AI behavior
- View and maintain dialog history
- Clear conversation history when needed
- User-friendly web interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   pip install "vllm>=0.4.3,<=0.7.3"
   ```
3. Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```
4. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   SECRET_KEY=your_random_secret_key_here
   ```

## Running the Application

Start the vllm server:

```
bash ./run_vllm_server.sh
```

Start the Flask server:

```
bash ./run_app.sh
```

Then open your browser and navigate to:
```
http://127.0.0.1:11001
```

## Note

This application uses Flask sessions to store conversation history, which means the history is tied to your browser session. Closing the browser or clearing cookies will reset the conversation. 
