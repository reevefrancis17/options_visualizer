#!/usr/bin/env python3
import os
import shutil
import sys

def clean_cache():
    """Clean the cache directory"""
    cache_dir = 'cache'
    
    # Check if the cache directory exists
    if os.path.exists(cache_dir):
        # Remove all files in the cache directory
        for file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        
        print("Cache cleaned successfully.")
    else:
        print(f"Cache directory {cache_dir} does not exist.")

if __name__ == "__main__":
    clean_cache() 