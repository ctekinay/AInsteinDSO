import os

def print_tree(directory, prefix="", ignore_dirs={'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}):
    """Print directory tree structure."""
    
    # Get all items in directory
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # Separate directories and files
    dirs = []
    files = []
    
    for item in items:
        if item.startswith('.') and item not in {'.gitignore', '.env'}:
            continue  # Skip hidden files except important ones
            
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            if item not in ignore_dirs:
                dirs.append(item)
        else:
            files.append(item)
    
    # Print directories first, then files
    all_items = dirs + files
    
    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item}")
        
        # Recurse into directories
        if item in dirs:
            extension = "    " if is_last else "│   "
            subdir_path = os.path.join(directory, item)
            print_tree(subdir_path, prefix + extension, ignore_dirs)

# Run it
print("Project Structure:")
print("=" * 50)
print_tree(".")  # Start from current directory