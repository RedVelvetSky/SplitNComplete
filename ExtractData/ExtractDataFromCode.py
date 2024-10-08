import os
import csv
import random
import argparse

def remove_invisible_characters(text):
    """
    Remove zero-width spaces and other invisible characters from the text.
    """

    invisible_chars = [
        '\u200B',  # Zero Width Space
        '\u200C',  # Zero Width Non-Joiner
        '\u200D',  # Zero Width Joiner
        '\uFEFF'   # Zero Width No-Break Space (BOM)
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    return text

def find_code_files(root_dir, extensions=('.py', '.cs')):
    """
    Recursively find all files in root_dir with the given extensions.
    """
    code_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extensions):
                code_files.append(os.path.join(subdir, file))
    return code_files

def generate_cursor_positions(lines, num_positions=5, middle_size=3):
    """
    Generate random cursor positions in the file.
    Ensures that there's enough lines for the middle.
    """
    max_pos = len(lines) - middle_size
    if max_pos <= 0:
        return []
    positions = random.sample(range(1, max_pos), min(num_positions, max_pos))
    return positions

def process_file(file_path, num_positions=5, middle_size=3):
    """
    Process a single file to generate code completion examples.
    Returns a list of tuples: (prefix, suffix, middle, filename)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [remove_invisible_characters(line) for line in lines]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    cursor_positions = generate_cursor_positions(lines, num_positions, middle_size)
    examples = []

    for pos in cursor_positions:
        prefix = ''.join(lines[:pos]).strip()
        middle = ''.join(lines[pos:pos + middle_size]).strip()
        suffix = ''.join(lines[pos + middle_size:]).strip()
        examples.append((prefix, suffix, middle, os.path.basename(file_path)))

    return examples

def write_csv(dataset, output_file):
    """
    Write the dataset to a CSV file.
    """
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
    parser = argparse.ArgumentParser(description="Generate code completion dataset from .py and .cs files.")
    parser.add_argument('repository_path', help='Path to the repository containing .py and .cs files')
    parser.add_argument('--output', default='code_completion_dataset.csv', help='Output CSV file name')
    parser.add_argument('--positions', type=int, default=20, help='Number of cursor positions per file')
    parser.add_argument('--middle_size', type=int, default=3, help='Number of lines to consider as the middle (completion)')
    args = parser.parse_args()

    code_files = find_code_files(args.repository_path)
    print(f"Found {len(code_files)} code files.")

    dataset = []
    for file in code_files:
        examples = process_file(file, args.positions, args.middle_size)
        dataset.extend(examples)
        print(f"Processed {file}: {len(examples)} examples.")

    print(f"Total examples collected: {len(dataset)}")
    write_csv(dataset, args.output)
    print(f"Dataset saved to {args.output}")

if __name__ == "__main__":
    main()
