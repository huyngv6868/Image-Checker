import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Setup OpenAI client for NVIDIA
api_key = os.getenv("NVIDIA_API_KEY")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

# Get all models
try:
    models = client.models.list()
    all_models = [m.id for m in models.data]
    print(f"Total models available: {len(all_models)}")
except Exception as e:
    print(f"Failed to list models: {e}")
    sys.exit(1)

# Filter for vision models (heuristics based on names)
vision_models = [m for m in all_models if 'vision' in m.lower() or 'vl' in m.lower() or 'paligemma' in m.lower() or 'deplot' in m.lower()]

# Some known ones just in case:
known_vision = [
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "microsoft/phi-3-vision-128k-instruct",
    "nvidia/nvlm-d-72b",
    "qwen/qwen2-vl-72b-instruct"
]
for km in known_vision:
    if km not in vision_models and km in all_models:
        vision_models.append(km)

print(f"Found {len(vision_models)} potential vision models: {vision_models}")

# Test image 1
img_path = "/Volumes/Huy/Stock-photos/Image-Packs/we.bond.creations/business-women-technical-discussion-tablet-office-556102384.jpeg"

# We will just patch vision.py MODEL_ID dynamically
sys.path.append(".")
import app.checks.vision as vision
import time

tested = 0
for model_id in vision_models[:10]:
    print(f"\n\n{'='*80}")
    print(f"Testing Model: {model_id}")
    vision.MODEL_ID = model_id
    
    try:
        results = vision.run_all(Path(img_path))
        for res in results:
            print(res)
    except Exception as e:
        print(f"Model {model_id} failed: {e}")
    
    # Sleep to avoid rate limits
    time.sleep(2)
    tested += 1
