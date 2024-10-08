import os
import csv
import argparse
import ast
import re
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


def find_code_files(root_dir: str, extensions: Tuple[str, ...] = ('.py', '.cs')) -> List[str]:
    """
    Recursively find all files in root_dir with the given extensions.
    """
    code_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions):
                code_files.append(os.path.join(subdir, file))
    return code_files


def parse_python_blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Parse Python file lines to identify function and class definitions.
    Returns a list of tuples indicating the start and end line numbers of each block.
    """
    code = ''.join(lines)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"SyntaxError while parsing Python file: {e}")
        return []

    blocks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1  # Convert to 0-based index
            if hasattr(node, 'end_lineno'):
                end = node.end_lineno
            else:
                # Fallback if end_lineno is not available (Python < 3.8)
                end = start + 1
            blocks.append((start, end))
    return blocks


def parse_csharp_blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Parse C# file lines to identify class and method definitions using regex.
    Returns a list of tuples indicating the start and end line numbers of each block.
    """
    blocks = []
    class_pattern = re.compile(r'^\s*(public|private|protected|internal)?\s*(class|struct|interface)\s+\w+')
    method_pattern = re.compile(r'^\s*(public|private|protected|internal|static|\s)+\s*\w+\s+\w+\s*\(.*\)\s*\{?')

    block_stack = []
    for idx, line in enumerate(lines):
        if class_pattern.match(line):
            block_stack.append(('class', idx))
        elif method_pattern.match(line):
            block_stack.append(('method', idx))
        elif '{' in line:
            if block_stack:
                block_type, start_idx = block_stack[-1]
                # Attempt to find the matching closing brace
                end_idx = find_matching_brace(lines, idx)
                if end_idx:
                    blocks.append((start_idx, end_idx))
                    block_stack.pop()
        elif '}' in line:
            if block_stack:
                block_stack.pop()
    return blocks


def find_matching_brace(lines: List[str], start_idx: int) -> Optional[int]:
    """
    Find the line index of the matching closing brace for a block starting at start_idx.
    """
    brace_count = 0
    for idx in range(start_idx, len(lines)):
        brace_count += lines[idx].count('{')
        brace_count -= lines[idx].count('}')
        if brace_count == 0:
            return idx + 1  # Return the line after the closing brace
    return None


def generate_cursor_positions(blocks: List[Tuple[int, int]], total_lines: int) -> List[int]:
    """
    Generate cursor positions between logical blocks to ensure completeness.
    """
    cursor_positions = []
    # Sort blocks by their start positions
    blocks = sorted(blocks, key=lambda x: x[0])
    # Cursor positions are set to the end of each block
    for block in blocks:
        end_line = block[1]
        if end_line < total_lines:
            cursor_positions.append(end_line)
    return cursor_positions


def process_python_file(file_path: str, num_positions: int = 5, middle_size: int = 3) -> List[
    Tuple[str, str, str, str]]:
    """
    Process a Python file to generate code completion examples.
    Returns a list of tuples: (file_path, language, prefix, suffix, middle)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    blocks = parse_python_blocks(lines)
    cursor_positions = generate_cursor_positions(blocks, len(lines))

    # Limit to num_positions
    cursor_positions = cursor_positions[:num_positions]

    examples = []
    for pos in cursor_positions:
        prefix = ''.join(lines[:pos]).strip()
        # Ensure pos + middle_size does not exceed total lines
        middle_end = min(pos + middle_size, len(lines))
        middle = ''.join(lines[pos:middle_end]).strip()
        suffix = ''.join(lines[middle_end:]).strip()
        examples.append((file_path, 'Python', prefix, suffix, middle))
    return examples


def process_csharp_file(file_path: str, num_positions: int = 5, middle_size: int = 3) -> List[
    Tuple[str, str, str, str]]:
    """
    Process a C# file to generate code completion examples.
    Returns a list of tuples: (file_path, language, prefix, suffix, middle)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    blocks = parse_csharp_blocks(lines)
    cursor_positions = generate_cursor_positions(blocks, len(lines))

    # Limit to num_positions
    cursor_positions = cursor_positions[:num_positions]

    examples = []
    for pos in cursor_positions:
        prefix = ''.join(lines[:pos]).strip()
        # Ensure pos + middle_size does not exceed total lines
        middle_end = min(pos + middle_size, len(lines))
        middle = ''.join(lines[pos:middle_end]).strip()
        suffix = ''.join(lines[middle_end:]).strip()
        examples.append((file_path, 'C#', prefix, suffix, middle))
    return examples


def process_file(file_path: str, num_positions: int = 5, middle_size: int = 3) -> List[Tuple[str, str, str, str, str]]:
    """
    Process a single file to generate code completion examples.
    Returns a list of tuples: (file_path, language, prefix, suffix, middle)
    """
    _, ext = os.path.splitext(file_path)
    if ext == '.py':
        return process_python_file(file_path, num_positions, middle_size)
    elif ext == '.cs':
        return process_csharp_file(file_path, num_positions, middle_size)
    else:
        return []


def write_csv(dataset: List[Tuple[str, str, str, str, str]], output_file: str):
    """
    Write the dataset to a CSV file.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'language', 'prefix', 'suffix', 'middle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in dataset:
            writer.writerow({
                'file_path': entry[0],
                'language': entry[1],
                'prefix': entry[2],
                'suffix': entry[3],
                'middle': entry[4]
            })


def main():
    parser = argparse.ArgumentParser(description="Generate code completion dataset from .py and .cs files.")
    parser.add_argument('repository_path', help='Path to the repository containing .py and .cs files')
    parser.add_argument('--output', default='code_completion_dataset.csv', help='Output CSV file name')
    parser.add_argument('--positions', type=int, default=5, help='Number of cursor positions per file')
    parser.add_argument('--middle_size', type=int, default=3,
                        help='Number of lines to consider as the middle (completion)')
    args = parser.parse_args()

    code_files = find_code_files(args.repository_path)
    print(f"Found {len(code_files)} code files.")

    dataset = []
    for file in tqdm(code_files, desc="Processing files"):
        examples = process_file(file, args.positions, args.middle_size)
        dataset.extend(examples)
        if examples:
            print(f"Processed {file}: {len(examples)} examples.")

    print(f"Total examples collected: {len(dataset)}")
    write_csv(dataset, args.output)
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()