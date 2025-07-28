#!/usr/bin/env python
"""Setup pipeline for product design year classification"""

import argparse
from pathlib import Path
import subprocess
import sys
import os


def run_step(script_name, args=[]):
    """Run a script step"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    script_path = script_dir / script_name

    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in {script_name}:")
        print(result.stderr)
        print(result.stdout)  # Also print stdout for more info
        return False
    print(result.stdout)  # Show the output from the script
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup data pipeline for product year classification')
    parser.add_argument('--skip-validation', action='store_true', help='Skip URL validation')
    parser.add_argument('--xlsx-path', default='../data/metadata/fetch_ALL.xlsx', help='Path to XLSX file')
    parser.add_argument('--output-dir', default='../data', help='Output directory')
    args = parser.parse_args()

    print("=== Product Design Year Classification Pipeline ===\n")

    # Step 1: Process XLSX
    print("Step 1: Processing product data...")
    if not run_step('process_xlsx.py'):
        return

    # Step 2: Validate URLs (optional)
    if not args.skip_validation:
        print("\nStep 2: Validating image URLs...")
        if not run_step('validate_urls.py'):
            print("URL validation failed, continuing anyway...")

    # Step 3: Create splits
    print("\nStep 3: Creating train/val/test splits...")
    if not run_step('create_splits.py'):
        return

    # Step 4: Analyze dataset
    print("\nStep 4: Analyzing dataset...")
    if not run_step('analyze_dataset.py'):
        print("Analysis failed, but pipeline complete")

    print("\n=== Pipeline Complete! ===")
    print(f"Your data is ready in {args.output_dir}/")
    print("\nNext steps:")
    print("1. Review the analysis in data/analysis/")
    print("2. Check data/metadata/processing_issues.txt for any problems")
    print("3. Start training with: python scripts/train.py --model_name efficientnet-b2")


if __name__ == "__main__":
    main()