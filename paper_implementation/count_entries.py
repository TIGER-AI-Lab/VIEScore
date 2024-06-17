import os
import json
import sys

# Check if a directory argument is provided
if len(sys.argv) != 2:
    print("Usage: python count.py <directory>")
    sys.exit(1)

# Use the provided directory argument
root_dir = sys.argv[1]

# Walk through the directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.json'):
            # Construct the file path
            file_path = os.path.join(subdir, file)
            try:
                # Open and load the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Count the number of entries
                    number_of_entries = len(data)
                    print(f'{file_path}: {number_of_entries} entries')
            except json.JSONDecodeError as e:
                print(f'Error reading {file_path}: {e}')
            except Exception as e:
                print(f'An error occurred with {file_path}: {e}')
