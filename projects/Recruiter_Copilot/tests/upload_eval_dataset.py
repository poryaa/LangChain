# tests/upload_eval_dataset.py

import json
import pathlib
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()  

DATASET_PATH = pathlib.Path(__file__).parent / "data" / "recruiter_copilot_eval_dataset.json"
DATASET_NAME = "recruiter_copilot_preproduction_evals"

client = Client()

# Load the JSON
with open(DATASET_PATH) as f:
    examples = json.load(f)

# Create the dataset in LangSmith (skip if already exists)
existing = [d.name for d in client.list_datasets()]
if DATASET_NAME in existing:
    print(f"Dataset '{DATASET_NAME}' already exists — skipping creation.")
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Pre-production eval dataset for the recruiter copilot, grounded in real resume content.",
    )
    print(f"Created dataset: {dataset.name} ({dataset.id})")

# Upload each example
inputs  = [{"input": ex["input"]} for ex in examples]
outputs = [
    {
        "expected_intent":           ex["expected_intent"],
        "expected_response_mode":    ex["expected_response_mode"],
        "expected_candidate_ids":    ex["expected_candidate_ids"],
        "expected_skills_mentioned": ex["expected_skills_mentioned"],
        "expected_answer_summary":   ex["expected_answer_summary"],
    }
    for ex in examples
]

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id,
)

print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.")