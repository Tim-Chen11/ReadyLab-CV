import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def analyze_dataset(splits_dir):
    """Create visualizations and analysis of the dataset"""

    # Load all splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        with open(Path(splits_dir) / f'{split_name}.json', 'r') as f:
            splits[split_name] = json.load(f)

    # Create analysis directory
    analysis_dir = Path(splits_dir).parent / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # 1. Decade distribution
    plt.figure(figsize=(12, 6))

    decade_data = []
    for split_name, split_data in splits.items():
        for entry in split_data:
            decade_data.append({
                'split': split_name,
                'decade': entry['decade'],
                'classification': entry['classification']
            })

    df = pd.DataFrame(decade_data)

    # Plot decade distribution
    plt.subplot(1, 2, 1)
    decade_counts = df.groupby(['split', 'decade']).size().unstack(fill_value=0)
    decade_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Images per Decade by Split')
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=0)
    plt.legend(title='Decade', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Classification distribution
    plt.subplot(1, 2, 2)
    top_classifications = df['classification'].value_counts().head(10).index
    df_top = df[df['classification'].isin(top_classifications)]
    class_counts = df_top.groupby(['split', 'classification']).size().unstack(fill_value=0)
    class_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Top 10 Product Classifications by Split')
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=0)
    plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(analysis_dir / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Products per decade over time
    plt.figure(figsize=(10, 6))

    all_data = []
    for split_data in splits.values():
        all_data.extend(split_data)

    # Count unique products per decade
    products_by_decade = {}
    for entry in all_data:
        decade = entry['decade']
        product_id = entry['product_id']
        if decade not in products_by_decade:
            products_by_decade[decade] = set()
        products_by_decade[decade].add(product_id)

    decades = sorted(products_by_decade.keys())
    product_counts = [len(products_by_decade[d]) for d in decades]

    plt.bar(decades, product_counts)
    plt.title('Number of Unique Products by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for i, (decade, count) in enumerate(zip(decades, product_counts)):
        plt.text(i, count + 1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(analysis_dir / 'products_per_decade.png', dpi=300)
    plt.close()

    # 4. Generate report
    report = []
    report.append("=== Dataset Analysis Report ===\n")

    # Overall statistics
    total_images = sum(len(split_data) for split_data in splits.values())
    unique_products = len(set(entry['product_id'] for split_data in splits.values() for entry in split_data))

    report.append(f"Total Images: {total_images}")
    report.append(f"Unique Products: {unique_products}")
    report.append(f"Average Images per Product: {total_images / unique_products:.2f}\n")

    # Decade coverage
    report.append("Decade Coverage:")
    for decade in sorted(products_by_decade.keys()):
        report.append(f"  {decade}: {len(products_by_decade[decade])} products")

    # Classification diversity
    all_classifications = [entry['classification'] for split_data in splits.values() for entry in split_data]
    unique_classifications = len(set(all_classifications))
    report.append(f"\nUnique Classifications: {unique_classifications}")

    # Country diversity
    all_countries = [entry['country'] for split_data in splits.values() for entry in split_data]
    unique_countries = len(set(all_countries))
    report.append(f"Unique Countries: {unique_countries}")

    # Save report
    with open(analysis_dir / 'dataset_report.txt', 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f"\nAnalysis saved to {analysis_dir}/")


if __name__ == "__main__":
    analyze_dataset('../data/splits')