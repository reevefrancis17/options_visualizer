#!/usr/bin/env python3
import os
import datetime

# Directories and files to ignore
IGNORE_DIRS = ['project_dump', '.git', '__pycache__', '.DS_Store', 'node_modules', '.venv', 'env', 'venv']
IGNORE_FILES = ['project_dump.py']
# File extensions that are likely to be binary
BINARY_EXTENSIONS = ['.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.dat', '.db', '.sqlite', '.sqlite3', '.jpg', '.jpeg', '.png', '.gif', '.ico', '.pdf']

def is_text_file(file_path):
    """Check if a file is a text file"""
    # Skip files with binary extensions
    if any(file_path.endswith(ext) for ext in BINARY_EXTENSIONS):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(1024)
            # Check for null bytes which indicate binary content
            if '\0' in sample:
                return False
            return True
    except Exception:
        return False

def dump_project():
    """Dump project structure and file contents to a text file"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create project_dump directory if it doesn't exist
    dump_dir = os.path.join(project_root, "project_dump")
    os.makedirs(dump_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output file with timestamp
    output_file = os.path.join(dump_dir, f"project_dump_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# PROJECT DUMP: {os.path.basename(project_root)}\n")
        f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # First, dump the project structure
        f.write("## PROJECT STRUCTURE\n\n")
        for root, dirs, files in os.walk(project_root):
            # Skip ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            # Get relative path
            rel_path = os.path.relpath(root, project_root)
            if rel_path == '.':
                rel_path = '/'
            
            # Write directory information
            f.write(f"{rel_path}\n")
            
            # List files in this directory
            for file_name in sorted(files):
                if file_name in IGNORE_FILES:
                    continue
                f.write(f"  - {file_name}\n")
        
        # Then, dump the file contents
        f.write("\n\n## FILE CONTENTS\n\n")
        for root, dirs, files in os.walk(project_root):
            # Skip ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file_name in sorted(files):
                if file_name in IGNORE_FILES:
                    continue
                
                file_path = os.path.join(root, file_name)
                rel_file_path = os.path.relpath(file_path, project_root)
                
                # Skip non-text files
                if not is_text_file(file_path):
                    f.write(f"\n### {rel_file_path} [BINARY FILE - CONTENT SKIPPED]\n")
                    continue
                
                # Write file information and content
                f.write(f"\n### {rel_file_path}\n")
                f.write("```\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                        content = file.read()
                        f.write(content)
                        if not content.endswith('\n'):
                            f.write('\n')
                except Exception as e:
                    f.write(f"[Error reading file: {str(e)}]\n")
                
                f.write("```\n")
    
    print(f"Project dump created: {output_file}")

if __name__ == "__main__":
    dump_project()