from datasets import load_dataset
import json
import random
from pathlib import Path


def main():
    print("Loading HotpotQA dataset...")
    ds = load_dataset("hotpot_qa", "distractor")

    # Get the validation set (which has questions)
    if "validation" in ds:
        data = ds["validation"]
    else:
        # Fallback to train if validation doesn't exist
        data = ds["train"]

    print(f"Dataset loaded with {len(data)} examples")

    # Pick 100 random indices
    total_examples = len(data)
    random_indices = random.sample(range(total_examples), min(100, total_examples))
    print(f"Selected {len(random_indices)} random examples")

    # First, let's inspect the structure of one example
    print("Inspecting data structure...")
    sample = data[0]
    print("Sample keys:", list(sample.keys()))
    print("Sample question:", sample.get("question", "N/A"))
    print("Sample answer:", sample.get("answer", "N/A"))
    if "context" in sample:
        print("Context type:", type(sample["context"]))
        print(
            "Context keys:",
            list(sample["context"].keys())
            if isinstance(sample["context"], dict)
            else "Not a dict",
        )
        print("Context:", sample["context"])

    # Transform to our format
    transformed_data = []
    for idx in random_indices:
        example = data[idx]

        # Transform the context format
        # HotpotQA has context as dict with 'title' and 'sentences' lists
        context = []
        if "context" in example and isinstance(example["context"], dict):
            titles = example["context"].get("title", [])
            sentences_list = example["context"].get("sentences", [])

            # Pair each title with its corresponding sentences
            for title, sentences in zip(titles, sentences_list):
                if isinstance(sentences, list):
                    text = " ".join(sentences)  # Join sentences into one text
                else:
                    text = str(sentences)
                context.append({"title": title, "text": text})
        else:
            # Handle unexpected format
            print(
                f"Warning: Unexpected context format for example {idx}: {type(example.get('context'))}"
            )
            continue

        # Determine difficulty (HotpotQA doesn't have this, so we'll assign based on context length)
        if len(context) <= 2:
            difficulty = "easy"
        elif len(context) <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Create our format
        transformed_example = {
            "qid": f"hotpot_{idx}",
            "difficulty": difficulty,
            "question": example["question"],
            "gold_answer": example["answer"],
            "context": context,
        }

        transformed_data.append(transformed_example)

    # Save to file
    output_file = Path("data/hotpot.json")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(transformed_data)} examples to {output_file}")

    # Show a sample
    if transformed_data:
        print("\nSample transformed example:")
        print(json.dumps(transformed_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
