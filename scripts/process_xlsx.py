import pandas as pd
import json
from pathlib import Path
import re
from collections import defaultdict


def parse_year(year_str):
    """Extract year/decade from various formats, limited to 1960s-2000s"""
    if pd.isna(year_str) or year_str == 'n.d.':
        return None

    year_str = str(year_str).strip()

    # Handle ranges like "1985-1990"
    if '-' in year_str:
        years = year_str.split('-')
        year_str = years[0]  # Take first year

    # Extract 4-digit year
    match = re.search(r'(19|20)\d{2}', year_str)
    if match:
        year = int(match.group(0))

        # Limit to 1960-2010 range
        if year <= 1960:
            return "1960s"
        elif year >= 2010:
            return "2000s"
        else:
            # Convert to decade normally
            decade = (year // 10) * 10
            return f"{decade}s"

    # Handle formats like "1980s"
    match = re.search(r'(19|20)\d0s?', year_str)
    if match:
        decade_str = match.group(0).rstrip('s')
        decade_num = int(decade_str)

        # Apply same limits
        if decade_num < 1960:
            return "1960s"
        elif decade_num > 2000:
            return "2000s"
        else:
            return f"{decade_str}s"

    return None

def process_product_data(xlsx_path, output_dir):
    """Process product design data from XLSX"""

    # Read XLSX
    df = pd.read_excel(xlsx_path)
    print(f"Loaded {len(df)} products from {xlsx_path}")

    # Process data
    processed_data = []
    issues = []
    decade_stats = defaultdict(int)

    for idx, row in df.iterrows():
        try:
            # Parse year to decade
            decade = parse_year(row['year'])
            if not decade:
                issues.append(f"Row {idx}: Invalid year '{row['year']}' for product '{row['name']}'")
                continue

            # Parse multiple image URLs
            image_urls = str(row['image_urls']).split('|||')
            image_urls = [url.strip() for url in image_urls if url.strip()]

            if not image_urls:
                issues.append(f"Row {idx}: No valid image URLs for '{row['name']}'")
                continue

            # Create entry for each image
            for img_idx, url in enumerate(image_urls):
                entry = {
                    'id': f"product_{idx:05d}_img_{img_idx}",
                    'product_id': f"product_{idx:05d}",
                    'url': url,
                    'decade': decade,
                    'year_raw': row['year'],
                    'name': row['name'],
                    'classification': row.get('classification', 'unknown'),
                    'makers': row.get('makers', 'unknown'),
                    'country': row.get('country', 'unknown'),
                    'image_index': img_idx,
                    'total_images': len(image_urls),
                    'dimension': row.get('dimension', ''),
                    'source': row.get('source', '')
                }

                processed_data.append(entry)
                decade_stats[decade] += 1

        except Exception as e:
            issues.append(f"Row {idx}: Error processing - {str(e)}")

    # Save processed metadata
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'processed_metadata.json', 'w') as f:
        json.dump(processed_data, f, indent=2)

    # Save issues
    if issues:
        with open(output_path / 'processing_issues.txt', 'w') as f:
            f.write('\n'.join(issues))

    # Generate comprehensive statistics
    stats = {
        'total_products': len(df),
        'total_images': len(processed_data),
        'products_with_valid_years': len(set(e['product_id'] for e in processed_data)),
        'issues_count': len(issues),
        'decades': dict(decade_stats),
        'classifications': {},
        'countries': {},
        'makers': {}
    }

    # Count by various fields
    for entry in processed_data:
        # Classification stats
        classification = entry['classification']
        stats['classifications'][classification] = stats['classifications'].get(classification, 0) + 1

        # Country stats
        country = entry['country']
        stats['countries'][country] = stats['countries'].get(country, 0) + 1

        # Maker stats
        makers = entry['makers']
        stats['makers'][makers] = stats['makers'].get(makers, 0) + 1

    with open(output_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Total products: {stats['total_products']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Products with valid years: {stats['products_with_valid_years']}")
    print(f"Processing issues: {stats['issues_count']}")
    print(f"\nImages per decade:")
    for decade in sorted(stats['decades'].keys()):
        print(f"  {decade}: {stats['decades'][decade]} images")

    return processed_data, stats


if __name__ == "__main__":
    process_product_data('../data/metadata/fetch_ALL.xlsx', '../data/metadata')