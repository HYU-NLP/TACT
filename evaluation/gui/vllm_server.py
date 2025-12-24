import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import torch.multiprocessing
from transformers import AutoTokenizer
import uvicorn
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=os.getenv("VLLM_MODEL_PATH"))
parser.add_argument("--port", type=int, default=int(os.getenv("VLLM_PORT", 11002)))
parser.add_argument("--max_tokens", type=int, default=int(os.getenv("VLLM_MAX_TOKENS", 4096)))
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = None
    stop: list[str] = None
    seed: int = None
    skip_special_tokens: bool = True

@app.on_event("startup")
def load_model():
    global tokenizer, engine
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
@app.post("/generate")
def generate(req: GenerationRequest):
    global tokenizer, engine

    logger.info("[Request]:")
    logger.info(f"\t [generate] Prompt (last 200 chars): {req.prompt[-200:]}")
    
    # Robust parameter handling
    temperature = req.temperature if req.temperature is not None and req.temperature > 0 else 0.7
    top_p = req.top_p if req.top_p is not None and 0 < req.top_p <= 1 else 0.9
    top_k = req.top_k if req.top_k is not None and req.top_k >= -1 else -1
    presence_penalty = req.presence_penalty if req.presence_penalty is not None else 0.0
    frequency_penalty = req.frequency_penalty if req.frequency_penalty is not None else 0.0
    repetition_penalty = req.repetition_penalty if req.repetition_penalty is not None and req.repetition_penalty > 0 else 1.0
    max_tokens = req.max_tokens if req.max_tokens is not None and req.max_tokens > 0 else args.max_tokens
    stop = req.stop if req.stop is not None else None
    seed = req.seed if req.seed is not None else None
    skip_special_tokens = req.skip_special_tokens if req.skip_special_tokens is not None else True

    input_ids = tokenizer(req.prompt, return_tensors="pt")["input_ids"][0].tolist()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id],
        skip_special_tokens=skip_special_tokens,
    )
    result = engine.generate([{"prompt_token_ids": input_ids}], sampling_params)[0]
    return {"response": result.outputs[0].text}
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    uvicorn.run("vllm_server:app", host="0.0.0.0", port=args.port, reload=False)
