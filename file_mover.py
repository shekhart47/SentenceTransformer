import os
import shutil
import glob
from pathlib import Path

def move_json_files(source_directory, destination_folder=“augmentation_set1”):
“””
Move all .json files from source directory to destination folder

```
Args:
    source_directory (str): Path to the directory containing .json files
    destination_folder (str): Name of the destination folder (default: "augmentation_set1")

Returns:
    dict: Summary of the operation
"""

# Convert to Path objects for easier handling
source_path = Path(source_directory)
dest_path = source_path / destination_folder

# Create destination folder if it doesn't exist
dest_path.mkdir(exist_ok=True)

# Find all .json files in the source directory
json_files = list(source_path.glob("*.json"))

if not json_files:
    print(f"No .json files found in {source_directory}")
    return {"moved": 0, "files": [], "errors": []}

moved_files = []
errors = []

print(f"Found {len(json_files)} .json files to move")
print(f"Destination: {dest_path}")
print("-" * 50)

for json_file in json_files:
    try:
        # Define destination file path
        dest_file = dest_path / json_file.name
        
        # Check if file already exists in destination
        if dest_file.exists():
            print(f"⚠️  File already exists: {json_file.name} - Skipping")
            continue
        
        # Move the file
        shutil.move(str(json_file), str(dest_file))
        moved_files.append(json_file.name)
        print(f"✅ Moved: {json_file.name}")
        
    except Exception as e:
        error_msg = f"❌ Error moving {json_file.name}: {str(e)}"
        errors.append(error_msg)
        print(error_msg)

# Summary
print("-" * 50)
print(f"Operation completed!")
print(f"Successfully moved: {len(moved_files)} files")
if errors:
    print(f"Errors encountered: {len(errors)} files")

return {
    "moved": len(moved_files),
    "files": moved_files,
    "errors": errors
}
```

def move_json_files_with_pattern(source_directory, pattern=”*.json”, destination_folder=“augmentation_set1”):
“””
Move files matching a specific pattern to destination folder

```
Args:
    source_directory (str): Path to the directory containing files
    pattern (str): File pattern to match (default: "*.json")
    destination_folder (str): Name of the destination folder

Returns:
    dict: Summary of the operation
"""

source_path = Path(source_directory)
dest_path = source_path / destination_folder

# Create destination folder if it doesn't exist
dest_path.mkdir(exist_ok=True)

# Find all files matching the pattern
matching_files = list(source_path.glob(pattern))

if not matching_files:
    print(f"No files matching '{pattern}' found in {source_directory}")
    return {"moved": 0, "files": [], "errors": []}

moved_files = []
errors = []

print(f"Found {len(matching_files)} files matching '{pattern}'")
print(f"Destination: {dest_path}")
print("-" * 50)

for file_path in matching_files:
    try:
        dest_file = dest_path / file_path.name
        
        if dest_file.exists():
            print(f"⚠️  File already exists: {file_path.name} - Skipping")
            continue
        
        shutil.move(str(file_path), str(dest_file))
        moved_files.append(file_path.name)
        print(f"✅ Moved: {file_path.name}")
        
    except Exception as e:
        error_msg = f"❌ Error moving {file_path.name}: {str(e)}"
        errors.append(error_msg)
        print(error_msg)

print("-" * 50)
print(f"Operation completed! Moved {len(moved_files)} files")

return {
    "moved": len(moved_files),
    "files": moved_files,
    "errors": errors
}
```

def list_json_files(directory):
“””
List all .json files in the specified directory

```
Args:
    directory (str): Path to the directory

Returns:
    list: List of .json file names
"""

json_files = list(Path(directory).glob("*.json"))

if json_files:
    print(f"JSON files found in {directory}:")
    for i, file_path in enumerate(json_files, 1):
        print(f"{i:2d}. {file_path.name}")
else:
    print(f"No .json files found in {directory}")

return [f.name for f in json_files]
```

# Example usage based on your screenshot structure

if **name** == “**main**”:
# Set your source directory path
# Based on your screenshot, it looks like you’re in: /datasets/datasets_augmented/
source_dir = “/datasets/datasets_augmented/”  # Update this path as needed

```
# Option 1: Use current working directory
# source_dir = os.getcwd()

# Option 2: Use relative path if script is in the same directory
# source_dir = "."

print("=== JSON File Mover ===")
print(f"Source directory: {source_dir}")

# First, list the JSON files to see what will be moved
print("\n1. Listing JSON files:")
json_file_list = list_json_files(source_dir)

if json_file_list:
    # Ask for confirmation
    response = input(f"\nDo you want to move {len(json_file_list)} JSON files to 'augmentation_set1' folder? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n2. Moving files:")
        result = move_json_files(source_dir)
        
        if result["errors"]:
            print("\nErrors encountered:")
            for error in result["errors"]:
                print(f"  {error}")
    else:
        print("Operation cancelled.")
else:
    print("No JSON files to move.")
```

# Alternative: Quick one-liner function

def quick_move_json_files():
“”“Quick function to move JSON files from current directory”””
current_dir = os.getcwd()
return move_json_files(current_dir)
