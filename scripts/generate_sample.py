import json, os

def generate_sample(num=50):
    samples = []
    for i in range(1, num+1):
        samples.append({
            "query_id": f"q{i}",
            "Eng_Query": f"What is the capital of France? {i}",
            "Eng_Answer": "Paris",
            "query_type": "DESCRIPTION",
            "target_lang": "hin",
            "passages": {
                "English_passages": ["Paris is the capital city of France.", "The Eiffel Tower is located in Paris."],
                "Translated_passages": [],
                "is_selected": [1, 0]
            }
        })
    return samples

if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "sample.json")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(generate_sample(50), f, indent=2, ensure_ascii=False)
    print(f"Generated {50} sample queries at {sample_path}")
