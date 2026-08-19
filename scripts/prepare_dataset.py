from datasets import load_dataset

# Load the official MSMARCO-XI dataset using streaming
ds = load_dataset(
    "ai4bharat/MSMARCO-XI",
    streaming=True
)

# Select only the first 10,000 records
train_data = ds["train"].take(10_000)

print("Selected 10,000 records for development.")