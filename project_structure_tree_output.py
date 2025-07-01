import os

EXCLUDE = {'.venv', '.idea', '__pycache__', '.git', 'node_modules', '.DS_Store'}

def print_tree(path='.', prefix='', level=0, max_depth=2):
    if level >= max_depth:
        return

    entries = sorted(e for e in os.listdir(path) if e not in EXCLUDE and not e.startswith('.'))
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = i == len(entries) - 1
        connector = '└── ' if is_last else '├── '
        print(prefix + connector + entry)
        if os.path.isdir(full_path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            print_tree(full_path, new_prefix, level + 1, max_depth)

if __name__ == '__main__':
    print("Project Structure:\n")
    print_tree('.', max_depth=3)  # You can change to 2 or 4 if needed
