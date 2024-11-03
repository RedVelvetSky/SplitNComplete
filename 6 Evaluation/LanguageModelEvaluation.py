import os

from openai import OpenAI
import pandas as pd
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
EXTENDED_DATASET_LOAD = os.getenv('EXTENDED_DATASET_FILE')
EVALUATION_OUTPUT = os.getenv('LLM_EVAL_FILE')


def evaluate_code_structured(original_code, generated_code):
    # function scheme to always get correct and determined response
    function_schema = {
        "name": "evaluateCode",
        "description": "Evaluate similarity and correctness between original and generated code. Return structured metrics, including similarity score, differences, and errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "similarity_score": {
                    "type": "integer",
                    "description": "Similarity score between 0 and 100",
                    "minimum": 0,
                    "maximum": 100
                },
                "differences": {
                    "type": "string",
                    "description": "List of differences and errors in generated code."
                },
                "error_count": {
                    "type": "integer",
                    "description": "Total number of errors found in the generated code."
                },
                "similarity_length_ratio": {
                    "type": "number",
                    "description": "Ratio of the generated code length to the original code length."
                }
            },
            "required": ["similarity_score", "differences", "error_count", "similarity_length_ratio"]
        }
    }

    # some simple prompt for llm
    prompt = f"""
    Compare the following original code and generated code. Provide a similarity score, list of differences, and an error count.

    **Original Code:**
    {original_code}

    **Generated Code:**
    {generated_code}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a professional assistant and experienced programmer who evaluates code similarity and correctness."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_schema],
            function_call={"name": "evaluateCode"},
            temperature=0.0,
            max_tokens=150
        )

        # standard response retrieving procedure
        function_call = response.choices[0].message.function_call
        function_arguments = function_call.arguments
        evaluation_data = json.loads(function_arguments)

        # here we need to extract each metric from json
        similarity_score = evaluation_data.get("similarity_score")
        differences = evaluation_data.get("differences", "").strip()
        error_count = evaluation_data.get("error_count", 0)
        similarity_length_ratio = evaluation_data.get("similarity_length_ratio", 0.0)

        return similarity_score, differences, similarity_length_ratio, error_count

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None, "Error during evaluation", None, None


df = pd.read_csv(EXTENDED_DATASET_LOAD)
gen_columns = [col for col in df.columns if col.startswith('gen_')]

# evaluating each row with tqdm to see information about progress
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating Code Similarity"):
    original_code = row['middle']
    for gen_col in gen_columns:
        generated_code = row[gen_col]

        if pd.isna(generated_code) or not generated_code.strip():
            df.at[index, f'eval_score_{gen_col}'] = None
            df.at[index, f'eval_diff_{gen_col}'] = "No generated code."
            df.at[index, f'similarity_length_ratio_{gen_col}'] = None
            df.at[index, f'error_count_{gen_col}'] = None
            continue

        # evaluation itself
        score, differences, similarity_length_ratio, error_count = evaluate_code_structured(original_code,
                                                                                            generated_code)

        df.at[index, f'eval_score_{gen_col}'] = score
        df.at[index, f'eval_diff_{gen_col}'] = differences
        df.at[index, f'similarity_length_ratio_{gen_col}'] = similarity_length_ratio
        df.at[index, f'error_count_{gen_col}'] = error_count

        # just in order not to violate OpenAI's rate limits too much
        time.sleep(1)

output_path = EVALUATION_OUTPUT
df.to_csv(output_path, index=False)
print(f"Evaluation results saved to {output_path}")
