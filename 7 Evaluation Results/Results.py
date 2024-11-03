import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

automatic_evaluation_results_path = os.getenv("AUTOMATIC_EVAL_FILE")
agregated_results_save_path = os.getenv("AGREGATED_RESULTS_FILE")

data = pd.read_csv(automatic_evaluation_results_path)

# models
models = [
    'gen_tiny_starcoder_py',
    'gen_starcoder2_3b',
    'gen_starcoder2_7b',
    'gen_starcoder2_15b'
]

# metrics
metrics = [
    'Levenshtein',
    'Cyclomatic_Complexity',
    'Cyclomatic_Complexity_Diff',
    'Cosine_Similarity'
]

# weights for each metric
weights = {
    'Levenshtein': 0.30,
    'Cyclomatic_Complexity': 0.15,
    'Cyclomatic_Complexity_Diff': 0.15,
    'Cosine_Similarity': 0.40
}

results = {}
normalized_metrics = {}
all_metrics_values = {metric: [] for metric in metrics}

for model in models:
    for metric in metrics:
        column_name = f'{model}_{metric}'
        if column_name in data.columns:
            metric_values = data[column_name].dropna()
            all_metrics_values[metric].extend(metric_values)

# min and max for each metric
metric_min_max = {}
for metric in metrics:
    values = all_metrics_values[metric]
    if values:
        metric_min_max[metric] = {'min': min(values), 'max': max(values)}
    else:
        metric_min_max[metric] = {'min': 0, 'max': 1}  # Default if empty

# avg and normalized metrics
for model in models:
    model_metrics = {}
    normalized_model_metrics = {}
    for metric in metrics:
        column_name = f'{model}_{metric}'
        if column_name in data.columns:
            metric_values = data[column_name].dropna()
            if not metric_values.empty:
                avg_value = metric_values.mean()
                model_metrics[metric] = avg_value
                min_val = metric_min_max[metric]['min']
                max_val = metric_min_max[metric]['max']
                if max_val - min_val == 0:
                    normalized_value = 0
                else:
                    if metric in ['Exact_Match', 'BLEU', 'ROUGE', 'Cosine_Similarity']:
                        normalized_value = (avg_value - min_val) / (max_val - min_val)
                    else:
                        normalized_value = (max_val - avg_value) / (max_val - min_val)
                normalized_model_metrics[metric] = normalized_value
            else:
                model_metrics[metric] = None
                normalized_model_metrics[metric] = None
        else:
            model_metrics[metric] = None
            normalized_model_metrics[metric] = None
    results[model] = model_metrics
    normalized_metrics[model] = normalized_model_metrics

results_df = pd.DataFrame(results).T
normalized_df = pd.DataFrame(normalized_metrics).T

# composite scores
composite_scores = {}
for model in models:
    score = 0
    total_weight = 0
    for metric in metrics:
        weight = weights.get(metric, 0)
        value = normalized_df.loc[model, metric]
        if pd.notna(value):
            score += weight * value
            total_weight += weight
    if total_weight > 0:
        composite_score = score / total_weight
    else:
        composite_score = 0
    composite_scores[model] = composite_score

results_df['Composite_Score'] = results_df.index.map(composite_scores)
results_df.to_csv(agregated_results_save_path)

# the best model is being determined here!
best_model = max(composite_scores, key=composite_scores.get)
print(f"The best model is: {best_model} with a composite score of {composite_scores[best_model]:.4f}")

print("\nAverage Metrics for Each Model:")
print(results_df)

# metrics for each model
plt.figure(figsize=(15, 8))
results_df[metrics].plot(kind='bar', figsize=(15, 8))
plt.title('Comparison of Star Coder models')
plt.xlabel('Model Name')
plt.ylabel('AVG Metric Score')
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# metric separately
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.title(f'Comparison of {metric} Across Models')
    plt.xlabel('Model')
    plt.ylabel(f'Average {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# composite scores
plt.figure(figsize=(8, 6))
sns.barplot(x=list(composite_scores.keys()), y=list(composite_scores.values()))
plt.title('Composite Scores of Models')
plt.xlabel('Model')
plt.ylabel('Composite Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()