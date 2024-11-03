import os
import csv
import random
import re
import ast
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

def remove_invisible_characters(text):
    invisible_chars = [
        '\u200B',  # Zero Width Space
        '\u200C',  # Zero Width Non-Joiner
        '\u200D',  # Zero Width Joiner
        '\uFEFF'   # Zero Width No-Break Space (BOM symbol)
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    return text

def find_code_files(root_dir, extensions=('.py', '.cs')) -> List[str]:
    code_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions):
                code_files.append(os.path.join(subdir, file))
    return code_files

def parse_python_functions(file_path: str, source_code: str) -> List[Tuple[int, int]]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []

    function_positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            for decorator in node.decorator_list:
                start_line = min(start_line, decorator.lineno - 1)

            if hasattr(node, 'end_lineno'):
                end_line = node.end_lineno
            else:
                end_line = node.lineno - 1
                for child_node in ast.walk(node):
                    if hasattr(child_node, 'lineno'):
                        end_line = max(end_line, child_node.lineno)
                end_line += 1

            function_positions.append((start_line, end_line))
    return function_positions

def process_python_file(file_path: str, num_positions: int = 5) -> List[Tuple[str, str, str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = remove_invisible_characters(f.read())
            lines = source_code.splitlines(keepends=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    function_positions = parse_python_functions(file_path, source_code)
    if not function_positions:
        return []

    selected_functions = random.sample(function_positions, min(num_positions, len(function_positions)))

    examples = []
    for start_line, end_line in selected_functions:
        prefix = ''.join(lines[:start_line]).strip()
        middle = ''.join(lines[start_line:end_line]).strip()
        suffix = ''.join(lines[end_line:]).strip()
        examples.append((prefix, suffix, middle, os.path.basename(file_path)))

    return examples

def process_cs_file(file_path: str, num_positions: int = 5) -> List[Tuple[str, str, str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = remove_invisible_characters(f.read())
            lines = source_code.splitlines(keepends=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    method_pattern = re.compile(
        r'(?P<signature>^\s*(public|private|protected|internal|static|\s)+\s+\w+\s+\w+\s*\(.*?\))\s*'
        r'(?P<body>\{(?:[^{}]*|\{(?:[^{}]*|\{[^{}]*\})*\})*\})',
        re.MULTILINE | re.DOTALL
    )

    method_positions = []
    for match in method_pattern.finditer(source_code):
        start_idx = match.start()
        end_idx = match.end()
        start_line = source_code[:start_idx].count('\n')
        end_line = source_code[:end_idx].count('\n') + 1
        method_positions.append((start_line, end_line))

    if not method_positions:
        return []

    selected_methods = random.sample(method_positions, min(num_positions, len(method_positions)))

    examples = []
    for start_line, end_line in selected_methods:
        prefix = ''.join(lines[:start_line]).strip()
        middle = ''.join(lines[start_line:end_line]).strip()
        suffix = ''.join(lines[end_line:]).strip()
        examples.append((prefix, suffix, middle, os.path.basename(file_path)))

    return examples

def process_file(file_path: str, num_positions: int = 5) -> List[Tuple[str, str, str, str]]:
    if file_path.endswith('.py'):
        return process_python_file(file_path, num_positions)
    elif file_path.endswith('.cs'):
        return process_cs_file(file_path, num_positions)
    else:
        return []

def write_csv(dataset: List[Tuple[str, str, str, str]], output_file: str):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['prefix', 'suffix', 'middle', 'filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in dataset:
            writer.writerow({
                'prefix': entry[0],
                'suffix': entry[1],
                'middle': entry[2],
                'filename': entry[3]
            })

def main():
    repository_path = os.getenv("REPOSITORY_DIR_PATH")
    output_file = os.getenv("OUTPUT_DATASET_FILE")
    num_positions = int(os.getenv("POSITIONS", 20))

    code_files = find_code_files(repository_path)
    print(f"Found {len(code_files)} code files.")

    dataset = []
    for file in code_files:
        examples = process_file(file, num_positions)
        dataset.extend(examples)
        print(f"Processed {file}: {len(examples)} examples.")

    print(f"Total number of examples collected: {len(dataset)}")
    write_csv(dataset, output_file)
    print(f"Dataset was saved to {output_file}")

if __name__ == "__main__":
    main()