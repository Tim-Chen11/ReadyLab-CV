#!/usr/bin/env python3
"""
Create small test splits for development and testing
"""
import json
import os
from pathlib import Path
import random


def create_small_splits(source_dir, output_dir, samples_per_decade=2):
    """
    Create small test splits with limited samples per decade

    Args:
        source_dir: Directory containing original splits
        output_dir: Directory to save test splits
        samples_per_decade: Number of samples per decade to include
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load original splits
    with open(os.path.join(source_dir, 'train.json'), 'r') as f:
        train_data = json.load(f)

    with open(os.path.join(source_dir, 'val.json'), 'r') as f:
        val_data = json.load(f)

    # Group by decade
    def group_by_decade(data):
        decades = {}
        for item in data:
            decade = item['decade']
            if decade not in decades:
                decades[decade] = []
            decades[decade].append(item)
        return decades

    train_by_decade = group_by_decade(train_data)
    val_by_decade = group_by_decade(val_data)

    # Sample limited items per decade
    small_train = []
    small_val = []

    for decade in ['1960s', '1970s', '1980s', '1990s', '2000s']:
        # Sample from train set
        if decade in train_by_decade:
            train_samples = random.sample(
                train_by_decade[decade],
                min(samples_per_decade, len(train_by_decade[decade]))
            )
            small_train.extend(train_samples)

        # Sample from val set
        if decade in val_by_decade:
            val_samples = random.sample(
                val_by_decade[decade],
                min(max(1, samples_per_decade // 2), len(val_by_decade[decade]))
            )
            small_val.extend(val_samples)

    # Shuffle the data
    random.shuffle(small_train)
    random.shuffle(small_val)

    # Save small splits
    with open(output_dir / 'train_small.json', 'w') as f:
        json.dump(small_train, f, indent=2)

    with open(output_dir / 'val_small.json', 'w') as f:
        json.dump(small_val, f, indent=2)

    print(f"Created small splits:")
    print(f"  Train: {len(small_train)} samples")
    print(f"  Val: {len(small_val)} samples")

    # Show distribution
    train_dist = {}
    val_dist = {}

    for item in small_train:
        decade = item['decade']
        train_dist[decade] = train_dist.get(decade, 0) + 1

    for item in small_val:
        decade = item['decade']
        val_dist[decade] = val_dist.get(decade, 0) + 1

    print(f"\nTrain distribution: {train_dist}")
    print(f"Val distribution: {val_dist}")

    return small_train, small_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create small test splits')
    parser.add_argument('--source_dir', type=str, default='data/splits',
                        help='Directory containing original splits')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Directory to save test splits')
    parser.add_argument('--samples_per_decade', type=int, default=2,
                        help='Number of samples per decade for training')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)

    create_small_splits(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        samples_per_decade=args.samples_per_decade
    )