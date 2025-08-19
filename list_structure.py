import os


def list_directory_structure(startpath, output_file):
    """
    Recursively lists directory structure and saves to a text file,
    excluding any folder named 'myenv'.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            # Remove 'myenv' from directories to traverse
            if '.venv' in dirs:
                dirs.remove('.venv')

            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)

            # Skip the myenv folder itself if it's the current root
            if os.path.basename(root) == '.venv':
                continue

            f.write('{}{}/\n'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write('{}{}\n'.format(subindent, file))


if __name__ == "__main__":
    # Get the directory where this script is located
    projects_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = "directory_structure.txt"

    print(f"Listing directory structure of: {projects_dir}")
    print(f"Excluding folders named '.venv'")
    print(f"Saving to: {output_filename}")

    list_directory_structure(projects_dir, output_filename)

    print("Done! Directory structure saved.")