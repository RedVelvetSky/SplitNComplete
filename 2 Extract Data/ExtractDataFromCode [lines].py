import os
import csv
import random
import re
from typing import List
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

def remove_comments_and_strings(source_code, language):
    if language == 'python':
        source_code = re.sub(r'#.*', '', source_code)
        source_code = re.sub(r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\"|\'[^\']*\'|\"[^\"]*\")', '', source_code)
    elif language == 'csharp':
        source_code = re.sub(r'//.*', '', source_code)
        source_code = re.sub(r'/\*[\s\S]*?\*/', '', source_code)
        source_code = re.sub(r'@"[^"]*"|\'[^\']*\'|\"[^\"]*\"', '', source_code)
    return source_code

def generate_cursor_positions(lines, num_positions=5, middle_size=3):
    min_prefix = 1
    min_suffix = 1
    valid_positions = []

    max_start = len(lines) - middle_size - min_suffix
    if max_start <= min_prefix:
        return []

    possible_positions = list(range(min_prefix, max_start + 1))
    positions = random.sample(possible_positions, min(num_positions, len(possible_positions)))
    return positions

def process_file(file_path, num_positions=5, middle_size=3):
    extension = os.path.splitext(file_path)[1]
    if extension == '.py':
        language = 'python'
    elif extension == '.cs':
        language = 'csharp'
    else:
        print(f"Unsupported file extension: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            source_code = remove_invisible_characters(source_code)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    code_without_comments = remove_comments_and_strings(source_code, language)
    lines = code_without_comments.splitlines(keepends=True)
    lines = [line for line in lines if line.strip()]

    if len(lines) < middle_size + 2:
        return []

    cursor_positions = generate_cursor_positions(lines, num_positions, middle_size)
    examples = []

    for pos in cursor_positions:
        prefix = ''.join(lines[:pos]).strip()
        middle = ''.join(lines[pos:pos + middle_size]).strip()
        suffix = ''.join(lines[pos + middle_size:]).strip()
        if not middle:
            continue
        filename = os.path.basename(file_path)
        examples.append((prefix, suffix, middle, filename))

    return examples

def write_csv(dataset, output_file):
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
    output_file = os.getenv("OUTPUT_DATASET_FILE", "code_completion_dataset.csv")
    num_positions = int(os.getenv("POSITIONS", 10))
    middle_size = int(os.getenv("MIDDLE_SIZE", 3))

    code_files = find_code_files(repository_path)
    print(f"Found {len(code_files)} code files.")

    dataset = []
    for file in code_files:
        examples = process_file(file, num_positions, middle_size)
        dataset.extend(examples)
        print(f"Processed {file}: {len(examples)} examples.")

    print(f"Total examples collected: {len(dataset)}")
    write_csv(dataset, output_file)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()