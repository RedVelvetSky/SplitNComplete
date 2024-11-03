import ast
import os
from collections import defaultdict
from difflib import unified_diff

import autopep8
import Levenshtein
import pandas as pd
from radon.complexity import cc_visit
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity  # Added: Import cosine_similarity

import sacrebleu
import h5py

nltk.download('punkt_tab', quiet=True)

SAVE_PATH = os.getenv("AUTOMATIC_EVAL_FILE")
DATASET_PATH = os.getenv("EXTENDED_DATASET_FILE")
EMBEDDINGS_PATH = os.getenv("OUTPUT_EMBEDDINGS_FILE")

def format_code(code):
    try:
        return autopep8.fix_code(code)
    except Exception as e:
        print(f"autopep8 formatting error: {e}")
    return code

# def structural_similarity(reference, candidate):
#     try:
#         ref_ast = ast.parse(reference)
#         cand_ast = ast.parse(candidate)
#
#         ref_dump = ast.dump(ref_ast)
#         cand_dump = ast.dump(cand_ast)
#
#         if ref_dump == cand_dump:
#             return {"similarity": 100, "differences": None, "exact_match": True}
#
#         common_nodes = sum(
#             1 for line in unified_diff(ref_dump.splitlines(), cand_dump.splitlines()) if line.startswith("  "))
#         total_nodes = max(len(ref_dump.splitlines()), len(cand_dump.splitlines()))
#         similarity_score = (common_nodes / total_nodes) * 100
#
#         differences = list(unified_diff(ref_dump.splitlines(), cand_dump.splitlines(), lineterm=""))
#
#         return {
#             "similarity": similarity_score,
#             "differences": differences,
#             "exact_match": False
#         }
#
#     except Exception as e:
#         return {"error": str(e), "similarity": 0, "differences": None, "exact_match": False}

def cyclomatic_complexity(code):
    try:
        analysis = cc_visit(code)
        total_complexity = sum([block.complexity for block in analysis])
        return total_complexity
    except Exception as e:
        return None

def exact_match(reference, candidate):
    if not isinstance(reference, str) or not isinstance(candidate, str):
        return 0
    return int(reference.strip() == candidate.strip())

def compute_bleu(reference, candidate):
    try:
        if not isinstance(reference, str) or not isinstance(candidate, str):
            return 0.0

        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)

        smoothie = SmoothingFunction().method4

        bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        return bleu_score
    except Exception as e:
        print(f"BLEU computation error: {e}")
        return 0.0

def compute_rouge(reference, candidate):
    try:
        if not isinstance(reference, str) or not isinstance(candidate, str):
            return 0.0

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)

        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"ROUGE computation error: {e}")
        return 0.0

def calculate_all_metrics(reference, candidate, h5f, index, model):
    """
    Calculate all metrics between reference and candidate, including cosine similarity.

    Parameters:
        reference (str): The reference code snippet.
        candidate (str): The candidate code snippet.
        h5f (h5py.File): Open HDF5 file object containing embeddings.
        index (int): The row index in the dataset.
        model (str): The model name corresponding to the candidate.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    metrics = {}

    # Format code if it's a string; else, use empty string
    reference_str = format_code(reference) if isinstance(reference, str) else ""
    candidate_str = format_code(candidate) if isinstance(candidate, str) else ""

    # Levenshtein
    levenshtein_distance = Levenshtein.distance(reference_str, candidate_str)
    max_len = max(len(reference_str), len(candidate_str))
    levenshtein_normalized = levenshtein_distance / max_len if max_len > 0 else 0

    # Cyclomatic Complexity
    cc_reference = cyclomatic_complexity(reference_str)
    cc_candidate = cyclomatic_complexity(candidate_str)
    cc_difference = (cc_reference or 0) - (cc_candidate or 0)

    # Exact Match Accuracy
    exact = exact_match(reference, candidate)

    # BLEU and ROUGE Metrics
    bleu = compute_bleu(reference_str, candidate_str)
    rouge = compute_rouge(reference_str, candidate_str)

    # Cosine Similarity
    try:
        middle_embedding = h5f[f"{index}/middle"][:] if f"{index}/middle" in h5f else None
        gen_embedding = h5f[f"{index}/{model}"][:] if f"{index}/{model}" in h5f else None

        if middle_embedding is not None and gen_embedding is not None:
            cosine_sim = cosine_similarity([middle_embedding], [gen_embedding])[0][0]
        else:
            cosine_sim = 0.0  # default val if embeddings are missing
    except Exception as e:
        print(f"Cosine similarity computation error: {e}")
        cosine_sim = 0.0

    # Metrics saving
    metrics['Levenshtein'] = levenshtein_normalized
    metrics['Cyclomatic_Complexity'] = cc_candidate
    metrics['Cyclomatic_Complexity_Diff'] = cc_difference
    metrics['Exact_Match'] = exact
    metrics['BLEU'] = bleu
    metrics['ROUGE'] = rouge
    metrics['Cosine_Similarity'] = cosine_sim

    return metrics

def get_embedding(h5f, index, column):
    dataset_key = f"{index}/{column}"
    if dataset_key in h5f:
        return h5f[dataset_key][:]
    else:
        return None

def calculate_cosine_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def main():
    df = pd.read_csv(DATASET_PATH)

    models = [col for col in df.columns if col.startswith('gen_')]
    metrics = {model: defaultdict(list) for model in models}
    middle_values = []

    with h5py.File(EMBEDDINGS_PATH, 'r') as h5f:
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
            reference = row['middle']

            # if NaN we treat it as empty string
            reference_str = format_code(str(reference)) if isinstance(reference, str) else ""

            middle_values.append(reference)

            for model in models:
                candidate = row[model]

                if pd.isna(candidate) or not isinstance(candidate, str):
                    # we need to handle NaNs for sure, append default metric values here
                    metrics[model]['Levenshtein'].append(0)
                    metrics[model]['Cyclomatic_Complexity'].append(0)
                    metrics[model]['Cyclomatic_Complexity_Diff'].append(0)
                    metrics[model]['Exact_Match'].append(0)
                    metrics[model]['BLEU'].append(0.0)
                    metrics[model]['ROUGE'].append(0.0)
                    metrics[model]['chrF'].append(0.0)
                    metrics[model]['Cosine_Similarity'].append(0.0)
                    continue

                # calling main func for calculating
                metric = calculate_all_metrics(reference_str, candidate, h5f, idx, model)

                metrics[model]['Levenshtein'].append(metric.get('Levenshtein', 0))
                metrics[model]['Cyclomatic_Complexity'].append(metric.get('Cyclomatic_Complexity', 0))
                metrics[model]['Cyclomatic_Complexity_Diff'].append(metric.get('Cyclomatic_Complexity_Diff', 0))
                metrics[model]['Exact_Match'].append(metric.get('Exact_Match', 0))
                metrics[model]['BLEU'].append(metric.get('BLEU', 0.0))
                metrics[model]['ROUGE'].append(metric.get('ROUGE', 0.0))
                metrics[model]['chrF'].append(metric.get('chrF', 0.0))
                metrics[model]['Cosine_Similarity'].append(metric.get('Cosine_Similarity', 0.0))

    # converting metrics to df
    metrics_df = pd.DataFrame(
        {f'{model}_{metric}': metrics[model][metric] for model in models for metric in metrics[model]}
    )
    metrics_df.insert(0, 'middle', middle_values)

    metrics_df.to_csv(SAVE_PATH, index=False)
    print(f"Metrics successfully saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()