#!/usr/bin/env python3
"""
Dash and unicode compliance checker.
Ensures zero em dashes or en dashes exist in comments, documentation, or code.
"""

import sys
import os

FORBIDDEN = [
    ('\u2014', 'em-dash (U+2014)'),
    ('\u2013', 'en-dash (U+2013)'),
]

ALLOWED_EXTENSIONS = {
    '.cl', '.clh', '.cpp', '.hpp', '.h', '.c',
    '.md', '.txt', '.py', '.sh', '.cmake', '.json',
    '.mermaid', '.yml', '.yaml'
}

def scan_file(filepath):
    errors = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for lineno, line in enumerate(f, 1):
                for char, name in FORBIDDEN:
                    if char in line:
                        errors.append(f"{filepath}:{lineno}: contains {name}")
    except Exception as e:
        errors.append(f"{filepath}: read error: {e}")
    return errors

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_errors = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.git' in dirpath or 'build' in dirpath:
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext in ALLOWED_EXTENSIONS or fname in ['Makefile', 'CMakeLists.txt', 'LICENSE']:
                full_path = os.path.join(dirpath, fname)
                errors = scan_file(full_path)
                all_errors.extend(errors)
                
    if all_errors:
        print(f"FAILED: Found {len(all_errors)} dash violations:")
        for err in all_errors:
            print("  " + err)
        sys.exit(1)
    else:
        print("PASS: Zero em-dash and en-dash violations detected across project.")
        sys.exit(0)

if __name__ == '__main__':
    main()
