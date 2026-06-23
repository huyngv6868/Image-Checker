from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(".")
from app.checks.vision import run_all

images = [
    "/Volumes/Huy/Stock-photos/Image-Packs/we.bond.creations/chic-woman-laughing-phone-coffee-urban-384756291.jpeg"
]

for img_path in images:
    print(f"\n\n{'='*80}")
    print(f"Testing {img_path}")
    results = run_all(Path(img_path))
    for res in results:
        print(res)
