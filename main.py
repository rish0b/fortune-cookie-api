from fastapi import FastAPI, HTTPException
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging
import random
import os
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# Retrieve env variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS")

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Change "*" to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# / root path
@app.get("/")
def root():
    return {"message": "Fortune Cookie 2.0"}

# /fortune API
@app.get("/fortune")
def read_fortune():
    return StreamingResponse(generate_fortune(), media_type="text/event-stream")

# Service method for streaming
def generate_fortune():
    try:
        # Strict response format
        response_format = {   
            "type": "json",
            "value": {
                "properties": {
                    "fortune": {"type": "string"},
                    "lucky_numbers": {"type": "array", "items": {"type": "integer"}, "size" : 6}
                },
                "required": ["fortune", "lucky_numbers"],
            }
        }

        # Tool calls
        topic = get_fortune_topic()
        depth = get_fortune_depth()
        lucky_numbers = get_lucky_numbers()

        # System and user prompts
        messages = [
            {
                "role": "system",
                "content": 
                "You are a fortune cookie generator. " + 
                "Your task is to create short and unique fortunes that could be found inside a fortune cookie. " + 
                "Each fortune should be short, concise, clear, and contain a positive message. " + 
                "Ensure the fortunes feel diverse and varied, avoiding repetition of phrases or themes. " +
                "Avoid using nature related themes excessively. " +
                "You have a limit of 75 tokens. " + 
                "Don't be overly metaphorical or philosophical. " +
                "You have a limit of 1 sentence, at the worst 2 sentences. " +
                "After the fortune, return a separate list of six lucky numbers between 1 and 100."
            },
            {
                "role": "user",
                "content": (
                    f"Generate a sensible wisdom about:\n"
                    f"Topic: {topic}\n"
                    f"With a relative depth and complexity of:\n"
                    f"Depth: {depth}\n"
                    f"Followed by a set of six lucky_numbers: {lucky_numbers}\n"
                )
            }
        ]

        # API call and stream data directly
        stream = client.chat_completion(            
            model=MODEL_NAME,
            messages=messages,
            temperature=1.3, 
            top_p=0.9,
            seed=random.randint(0, 10000),
            max_tokens=100,
            frequency_penalty=0.5,
            response_format=response_format,
            stream=True
        )

        # Streaming content back chunk by chunk
        fortune_parts = []  
        for token in stream:
            content = token.choices[0].delta.content 
            fortune_parts.append(content) 
            yield f"{content}"  

        # Log the completed fortune after streaming
        complete_fortune = " ".join(fortune_parts)
        logger.info(f"Completed fortune: {complete_fortune}")

    except Exception as e:
        # Log the exception
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Oops! The fortunes took a wrong turn. Try again!")

# Arbitrary fortune topic
def get_fortune_topic():
    topics = [
        "Personal Growth", 
        "Love and Relationships", 
        "Work and Career", 
        "Luck and Fortune", 
        "Nature and the Environment", 
        "Creativity and Innovation", 
        "Wisdom and Philosophy", 
        "Health and Well-being", 
        "Travel and Adventure", 
        "Change and Transformation", 
        "Wealth and Abundance", 
        "Success and Achievement", 
        "Mindset and Perspective", 
        "Spirituality", 
        "Serendipity and Luck",
        "Romance and Love"
    ]
    return random.choice(topics)

# Retrieve fortune depth
def get_fortune_depth():
    depth = [
        "Minimal",
        "Shallow",
        "Moderate",
        "Deep"
        ]
    return random.choice(depth)

# Lucky number generation with cultural skew
def get_lucky_numbers():
    ranges = [ {"range": range(1, 11), "weight": 5},  {"range": range(11, 50), "weight": 1}, {"range": range(50, 100), "weight": 1}, ]
    favored_numbers = {8: 3, 7: 3, 88: 5}  
    total_range_weight = sum(r["weight"] * len(r["range"]) for r in ranges)
    total_favored_weight = sum(favored_numbers.values())
    total_weight = total_range_weight + total_favored_weight

    def generate_lucky_number():
        pick = random.uniform(0, total_weight)
        cumulative_weight = 0
        for number, weight in favored_numbers.items():
            cumulative_weight += weight
            if pick <= cumulative_weight:
                return number

        for r in ranges:
            range_weight = r["weight"] * len(r["range"])
            cumulative_weight += range_weight
            if pick <= cumulative_weight:
                return random.choice(r["range"])

    lucky_numbers = set()
    while len(lucky_numbers) < 6:
        lucky_numbers.add(generate_lucky_number())
    
    return list(lucky_numbers)
