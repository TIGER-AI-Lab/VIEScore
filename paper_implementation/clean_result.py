import json
import os
from typing import Union, List, Optional
import sys

def get_file_path(filename: Union[str, os.PathLike], search_from: Union[str, os.PathLike] = "."):
    """
    Search for a file across a directory and return its absolute path.

    Args:
        filename (Union[str, os.PathLike]): The name of the file to search for.
        search_from (Union[str, os.PathLike], optional): The directory from which to start the search. Defaults to ".".

    Returns:
        str: Absolute path to the found file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    for root, dirs, files in os.walk(search_from):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")

def delete_entries_from_json(input_file, specific_words, overwrite=False):
    """
    Deletes entries from a JSON file that contain any of the specified words, prints out each deletion,
    and finally prints the total number of deletions and the remaining keys. 
    The file is either overwritten or saved as 'modified.json' based on the overwrite parameter.

    :param input_file: Path to the input JSON file.
    :param specific_words: List of words based on which entries are deleted.
    :param overwrite: Boolean indicating whether to overwrite the original file. Defaults to True.
    """
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Initialize deletion counter
    deleted_count = 0
    
    # Identify keys to delete and the matching word
    for key, value in list(data.items()):
        if 'score' not in value:  # Check if 'score' key is missing
            print(f"Deleting key '{key}' as it does not contain 'score'.")
            del data[key]
            deleted_count += 1
            continue
        elif 'score' not in value:  # Check if 'score' key is here but...
            if isinstance(data['score'], str):
                print(f"Deleting key '{key}' as it contain str in 'score'.")
                del data[key]
                deleted_count += 1
                continue
        if 'reasoning' not in value: # Check if 'reasoning' key is missing
            pass

        entry_as_string = json.dumps(value)

        for word in specific_words:
            if word in entry_as_string.lower():
                print(f"Deleting key '{key}' as it matches the specific word '{word}'.")
                del data[key]
                deleted_count += 1
                break  # Break to avoid multiple deletions for the same key

    # Determine the output file name
    output_file = input_file if overwrite else 'modified.json'

    # Write the modified dictionary back to the specified file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    # Print summary
    print(f"Total keys deleted: {deleted_count}")
    print(f"Remaining keys in the file: {len(list(data.keys()))}")
    print(f"File '{output_file}' has been saved with the modifications.")

# Open the file and read the lines
with open(get_file_path('banned_reasonings.txt'), 'r') as file:
    lines = file.readlines()
# Removing any trailing newlines or whitespace
banned_words = [line.strip() for line in lines]

def clean_json_files(root_dir):
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.json'):
                # Construct full file path
                file_path = os.path.join(dirpath, file)
                print(f"Cleaning file: {file_path}")
                delete_entries_from_json(file_path, banned_words, overwrite=True)

# Check if a directory argument is provided
if len(sys.argv) != 2:
    print("Usage: python count.py <directory>")
    sys.exit(1)

# Use the provided directory argument
selected_dir = sys.argv[1]

clean_json_files(selected_dir)