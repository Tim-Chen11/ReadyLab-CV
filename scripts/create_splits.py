import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from pathlib import Path


def create_product_aware_splits(metadata_path, output_dir,
                                val_ratio=0.15, test_ratio=0.15,
                                min_images_per_decade=50):
    """
    Create splits ensuring:
    1. All images of same product stay together
    2. Balanced decades in each split
    3. Stratified by classification if possible
    """

    # Load metadata
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # Group by product_id to keep product images together
    products = defaultdict(list)
    for entry in data:
        products[entry['product_id']].append(entry)

    # Analyze products by decade and classification
    product_info = {}
    decade_products = defaultdict(list)

    for product_id, images in products.items():
        # All images of a product have same metadata
        first_image = images[0]
        product_info[product_id] = {
            'decade': first_image['decade'],
            'classification': first_image['classification'],
            'country': first_image['country'],
            'images': images
        }
        decade_products[first_image['decade']].append(product_id)

    # Check if we have enough data
    print("\nProducts per decade:")
    for decade, product_ids in sorted(decade_products.items()):
        image_count = sum(len(products[pid]) for pid in product_ids)
        print(f"  {decade}: {len(product_ids)} products, {image_count} images")

    # Split products (not images) to avoid leakage
    train_products = []
    val_products = []
    test_products = []

    # Stratified split by decade
    for decade, product_ids in decade_products.items():
        if len(product_ids) < 3:
            # Too few products, put all in train
            train_products.extend(product_ids)
            print(f"Warning: {decade} has only {len(product_ids)} products, all going to train")
        else:
            # First split off test set
            train_val_ids, test_ids = train_test_split(
                product_ids,
                test_size=test_ratio,
                random_state=42
            )

            # Then split train and val
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=val_ratio / (1 - test_ratio),
                random_state=42
            )

            train_products.extend(train_ids)
            val_products.extend(val_ids)
            test_products.extend(test_ids)

    # Convert back to image entries
    train_data = []
    val_data = []
    test_data = []

    for pid in train_products:
        train_data.extend(product_info[pid]['images'])
    for pid in val_products:
        val_data.extend(product_info[pid]['images'])
    for pid in test_products:
        test_data.extend(product_info[pid]['images'])

    # Shuffle images within each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        # Save full metadata
        with open(Path(output_dir) / f'{split_name}.json', 'w') as f:
            json.dump(split_data, f, indent=2)

        # Save URL list for easy downloading
        urls = [d['url'] for d in split_data]
        with open(Path(output_dir) / f'{split_name}_urls.txt', 'w') as f:
            f.write('\n'.join(urls))

    # Print detailed statistics
    print(f"\n=== Split Statistics ===")
    print(f"Train: {len(train_products)} products, {len(train_data)} images")
    print(f"Val: {len(val_products)} products, {len(val_data)} images")
    print(f"Test: {len(test_products)} products, {len(test_data)} images")

    # Per-decade breakdown
    for split_name, split_data in splits.items():
        print(f"\n{split_name.capitalize()} split decades:")
        decade_counts = defaultdict(int)
        for entry in split_data:
            decade_counts[entry['decade']] += 1
        for decade in sorted(decade_counts.keys()):
            print(f"  {decade}: {decade_counts[decade]} images")

    # Save split summary
    summary = {
        'train_products': len(train_products),
        'val_products': len(val_products),
        'test_products': len(test_products),
        'train_images': len(train_data),
        'val_images': len(val_data),
        'test_images': len(test_data),
        'splits_by_decade': {}
    }

    for split_name, split_data in splits.items():
        decade_counts = defaultdict(int)
        for entry in split_data:
            decade_counts[entry['decade']] += 1
        summary['splits_by_decade'][split_name] = dict(decade_counts)

    with open(Path(output_dir) / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    create_product_aware_splits(
        '../data/metadata/processed_metadata.json',
        '../data/splits'
    )