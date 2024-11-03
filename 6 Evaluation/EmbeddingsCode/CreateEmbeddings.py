import os

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

dataset_path = os.getenv('EXTENDED_DATASET_FILE')
embeddings_save_path = os.getenv('OUTPUT_EMBEDDINGS_FILE')

# embedding generation function
def get_openai_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:30]}. Error: {e}")
        return None

# normalization
def normalize_embedding(embedding):
    embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print(df.head())

# columns to embed (middle and all gen_* columns)
embedding_columns = ['middle'] + [col for col in df.columns if col.startswith('gen_')]
print(f"Columns to embed: {embedding_columns}")

# init HDF5 storage. I think it's better choice for embeddings that csv or json anyway
with h5py.File(embeddings_save_path, 'w') as h5f:
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating EmbeddingsCode"):
        for col in embedding_columns:
            text = row[col]
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                continue

            embedding = get_openai_embedding(text)
            if embedding is not None:
                embedding = normalize_embedding(embedding)
                h5f.create_dataset(f"{index}/{col}", data=embedding)
            else:
                print(f"Failed to generate embedding for index {index}, column {col}")

print(f"EmbeddingsCode saved to {embeddings_save_path}")