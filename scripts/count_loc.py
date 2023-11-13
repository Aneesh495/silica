#!/usr/bin/env python3
"""
Accurate lines-of-code counter for Silica.
Measures OpenCL kernels (.cl, .clh), C++ host code (.cpp, .hpp), tests, and docs.
"""

import os
import sys

def count_lines(filepath):
    total = 0
    blank = 0
    comment = 0
    code = 0
    in_block_comment = False
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                total += 1
                stripped = line.strip()
                if not stripped:
                    blank += 1
                    continue
                if in_block_comment:
                    comment += 1
                    if '*/' in stripped:
                        in_block_comment = False
                    continue
                if stripped.startswith('/*'):
                    comment += 1
                    if '*/' not in stripped:
                        in_block_comment = True
                    continue
                if stripped.startswith('//') or stripped.startswith('#'):
                    comment += 1
                    continue
                code += 1
    except Exception:
        pass
    return total, code, comment, blank

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    categories = {
        'OpenCL Kernels (.cl, .clh)': {'extensions': ('.cl', '.clh'), 'dirs': ('kernels',), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
        'C++ Host Headers (.hpp)': {'extensions': ('.hpp',), 'dirs': ('include',), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
        'C++ Host Sources (.cpp)': {'extensions': ('.cpp',), 'dirs': ('src',), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
        'Tests (.cpp, .hpp)': {'extensions': ('.cpp', '.hpp'), 'dirs': ('tests',), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
        'Benchmarks (.cpp, .hpp)': {'extensions': ('.cpp', '.hpp'), 'dirs': ('benchmarks',), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
        'Documentation (.md)': {'extensions': ('.md',), 'dirs': ('docs', 'README.md'), 'total': 0, 'code': 0, 'comment': 0, 'blank': 0, 'files': 0},
    }
    
    for dirpath, _, filenames in os.walk(root):
        if '.git' in dirpath or 'build' in dirpath:
            continue
        relpath = os.path.relpath(dirpath, root)
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            full_file = os.path.join(dirpath, fname)
            rel_file = os.path.relpath(full_file, root)
            
            for cat_name, cat_data in categories.items():
                match_dir = False
                for d in cat_data['dirs']:
                    if rel_file.startswith(d) or relpath.startswith(d):
                        match_dir = True
                        break
                if match_dir and ext in cat_data['extensions']:
                    t, c, cm, b = count_lines(full_file)
                    cat_data['total'] += t
                    cat_data['code'] += c
                    cat_data['comment'] += cm
                    cat_data['blank'] += b
                    cat_data['files'] += 1
                    break

    print("=" * 76)
    print(f"{'Category':<32} {'Files':>6} {'Total Lines':>12} {'Code':>10} {'Comments':>10}")
    print("=" * 76)
    grand_total = 0
    grand_code = 0
    grand_opencl = 0
    
    for cat_name, d in categories.items():
        print(f"{cat_name:<32} {d['files']:>6} {d['total']:>12} {d['code']:>10} {d['comment']:>10}")
        grand_total += d['total']
        grand_code += d['code']
        if 'OpenCL' in cat_name:
            grand_opencl = d['total']
            
    print("=" * 76)
    print(f"{'GRAND TOTAL':<32} {'':>6} {grand_total:>12} {grand_code:>10}")
    print(f"OpenCL Total Lines: {grand_opencl}")
    print("=" * 76)

if __name__ == '__main__':
    main()
