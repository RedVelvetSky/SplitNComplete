import pandas as pd
from prompt_toolkit.key_binding import KeyBindings
from rich.columns import Columns
from rich.console import Console
from rich.syntax import Syntax
from rich.prompt import IntPrompt
from rich.panel import Panel
import os
import csv
import sys
from dotenv import load_dotenv

load_dotenv()

console = Console()

# metrics for evaluation
METRICS = [
    "Correctness",
    "Readability",
    "Efficiency",
    "Style",
    "Overall Quality"
]

evaluation_data = []

DATASET_PATH = os.getenv('EXTENDED_DATASET_FILE')
RESULTS_PATH = os.getenv('HUMAN_EVAL_FILE')

# we need key bindings to be able to escape freely if we tried of evaluating without loosing progress :)
bindings = KeyBindings()
@bindings.add('c-c')
@bindings.add('q')
def exit_application(event=None):
    console.print("[yellow]Exiting and saving all evaluations...[/yellow]")
    save_all_evaluations()
    sys.exit(0)

def load_dataset(path):
    try:
        df = pd.read_csv(path, sep=',')
        return df
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        sys.exit(1)

def initialize_results(path, models):
    if not os.path.exists(path):
        headers = ['filename', 'model'] + METRICS
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def save_evaluation(filename, model, ratings):
    evaluation_data.append([filename, model] + ratings)

def save_all_evaluations():
    if evaluation_data:
        with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(evaluation_data)
        evaluation_data.clear()

def prompt_ratings(metrics):
    ratings = []
    for metric in metrics:
        while True:
            try:
                rating = IntPrompt.ask(f"Rate the [bold]{metric}[/bold] (1-5)", choices=["1", "2", "3", "4", "5"])
                ratings.append(rating)
                break
            except Exception:
                console.print("[red]Invalid input. Please enter a number between 1 and 5.[/red]")
    return ratings

def display_code(original, generated, filename, model):
    syntax_original = Syntax(original, "python", theme="monokai", line_numbers=True, word_wrap=True)
    syntax_generated = Syntax(generated, "python", theme="monokai", line_numbers=True, word_wrap=True)

    panel_original = Panel(syntax_original, title="Original Code", border_style="cyan")
    panel_generated = Panel(syntax_generated, title="Generated Code", border_style="green")

    console.print(f"\n[bold magenta]File: {filename} | Model: {model}[/bold magenta]")
    console.print(Columns([panel_original, panel_generated]))

def main():
    df = load_dataset(DATASET_PATH)
    model_columns = [col for col in df.columns if col.startswith('gen_')]
    if not model_columns:
        console.print("[red]No generated model columns found in the dataset.[/red]")
        sys.exit(1)

    initialize_results(RESULTS_PATH, model_columns)
    total_entries = len(df) * len(model_columns)
    entry_counter = 1

    for row in df.itertuples(index=False):
        filename = row.filename
        original_code = row.middle

        for model in model_columns:
            generated_code = getattr(row, model)

            console.clear()
            console.rule(f"Entry {entry_counter}/{total_entries} | Model: {model}")

            if pd.isna(generated_code):
                console.print("[yellow]Generated code is missing for this entry.[/yellow]")
                entry_counter += 1
                continue

            display_code(original_code, generated_code, filename, model)
            console.print("\n[bold]Please rate the generated code based on the following metrics:[/bold]")
            ratings = prompt_ratings(METRICS)

            save_evaluation(filename, model, ratings)
            console.print("[green]Ratings saved successfully![/green]")
            console.print("[bold cyan]Press Enter to continue to the next entry...[/bold cyan]")
            input()
            entry_counter += 1

    save_all_evaluations()
    console.print("[bold green]Evaluation completed![/bold green]")

if __name__ == "__main__":
    main()