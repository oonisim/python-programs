#!/usr/bin/env python3
"""Update test files to use tempfile.TemporaryDirectory() and rename base_dir to result_dir."""

import re
from pathlib import Path

def update_test_file(filepath):
    """Update a single test file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # 1. Add tempfile import if not present
    if 'import tempfile' not in content:
        # Add after other imports
        import_pattern = r'(import torch\n)'
        content = re.sub(import_pattern, r'import tempfile\n\1', content)

    # 2. Replace base_dir with result_dir
    content = content.replace('base_dir', 'result_dir')

    # 3. For each test function, wrap with tempfile context and change "/tmp" to tmpdir
    def wrap_test_function(match):
        indent = match.group(1)
        func_def = match.group(2)
        func_body = match.group(3)

        # Replace "/tmp" with tmpdir in the function body
        func_body_updated = func_body.replace('result_dir="/tmp"', 'result_dir=tmpdir')

        # Add tempfile context manager
        return (f'{indent}{func_def}\n'
                f'{indent}    with tempfile.TemporaryDirectory() as tmpdir:\n'
                f'{func_body_updated}')

    # Find all test functions and wrap them
    # Match: def test_xxx():\n followed by the body (indented with 4 spaces)
    pattern = r'(^)(def test_\w+\(\):.*?\n)((?:    .*\n)+)'
    content = re.sub(pattern, wrap_test_function, content, flags=re.MULTILINE)

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    else:
        print(f"No changes: {filepath}")
        return False

if __name__ == '__main__':
    test_dir = Path(__file__).parent.parent / 'test'
    test_files = [
        'test_callback_integration.py',
        'test_callback_timing.py',
        'test_callbacks.py',
        'test_early_stopping.py',
        'test_memory_usage.py',
        'test_snapshot_persistence.py',
    ]

    for test_file in test_files:
        filepath = test_dir / test_file
        if filepath.exists():
            update_test_file(filepath)
