import os
import h5py
from dotenv import load_dotenv

load_dotenv()

embeddings_save_path = os.getenv("OUTPUT_EMBEDDINGS_FILE")

with h5py.File(embeddings_save_path, 'r') as h5f:
    # I want to check all stored embeddings keys just to verify
    print("Stored embeddings keys:", list(h5f.keys()))

    # Simple loop through a few samples to output
    for index in range(5):
        for col in ['middle', 'gen_tiny_starcoder_py', 'gen_starcoder2_3b', 'gen_starcoder2_7b', 'gen_starcoder2_15b']:
            dataset_key = f"{index}/{col}"
            if dataset_key in h5f:
                embedding = h5f[dataset_key][:]
                print(f"Embedding for row {index}, column '{col}':", embedding)
            else:
                print(f"No embedding found for row {index} and column '{col}' :(")
