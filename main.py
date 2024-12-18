from fastapi import FastAPI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import random
import os

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)

app = FastAPI()

@app.get("/fortune")
def read_fortune():
    return get_fortune()

def get_fortune() -> str:
    try:
        prompt = "Generate ONE random, short fortune-cookie fortune followed by a set of six lucky numbers between 1 and 100. No coding, no instructions. Just the single fortune and the array of six lucky numbers between 1 and 100."
        grammar={   
            "type": "json",
            "value": {
                "properties": {
                    "fortune": {"type": "string"},
                    "lucky_numbers": {"type": "array", "items": {"type": "integer"}, "size" : 6}
                },
                "required": ["fortune", "lucky_numbers"],
            }
        }
        return client.text_generation(
            prompt=prompt, 
            return_full_text=False, 
            temperature=0.7, 
            top_p=0.9,
            seed=random.randint(0, 10000), 
            repetition_penalty=1.9, 
            details=True, 
            grammar=grammar
        ) ["generated_text"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_lucky_numbers():
    return sorted(random.sample(range(1, 61), 6)) 