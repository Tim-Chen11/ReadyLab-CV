import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from gql import Client
import pandas as pd
from collections import Counter
from src.datacollection.design_object_model import DesignObject
from src.datacollection.fetch_cooper_hewitt import create_client, fetch_design_objects

def fetch_from_cooper_hewitt() -> int:
    # # Countries to consider for Cooper Hewitt
    # AMERICA_CANADA_COUNTRIES = [
    #     "USA",
    #     "U.S.A.",
    #     "USA (silver)",
    #     "USA or England",
    #     "USA or Europe",
    #     "United States",
    #     "Puerto Rico",
    #     "possibly USA",
    #     "probably USA",
    #     "Canada",
    # ]
    # department = "Product Design and Decorative Arts"
    # yearRange = range(1960, 2010)
    # size = 100
    # page = 0
    #
    # client = create_client()
    #
    # total_count = 0
    #
    # for year in yearRange:
    #     year_count = 0
    #
    #     for country in AMERICA_CANADA_COUNTRIES:
    #         results = fetch_design_objects(client, department, year, country, size, page)
    #         count = len(results)
    #         year_count += count
    #         total_count += count
    #
    #     print(f"Year: {year}, Found: {year_count}")
    #
    # print(f"\nTotal objects found: {total_count}")
    # return total_count
    return 0



def fetch_from_MoMA() -> int:
    # Countries to consider for MoMA in Artworks.csv file
    # No longer consider, as we do web scrap directly
    # USA = {
    #     "American",
    #     "American, born Eritrea",
    #     "American, born Mexico.",
    #     "Native American",
    # }
    # CANADA = {
    #     "Canadian",
    #     "Canadian Inuit",
    #     "Member of Wood Mountain Lakota First Nations",
    #     "Oneida",
    #     "Spirit Lake Dakota/Cheyenne River Lakota",
    # }



    return 0

def fetch_from_1stdibs() -> int:
    urls_for_fetching = {
        "https://www.1stdibs.com/furniture/?origin=american,canadian&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest",
        "https://www.1stdibs.com/jewelry/?origin=american,canadian&page=9&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest",
        "https://www.1stdibs.com/fashion/handbags-purses-bags/?origin=american,canadian&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest",
        "https://www.1stdibs.com/fashion/clothing/shoes/?origin=american,canadian&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest",
        "https://www.1stdibs.com/fashion/accessories/?origin=american,canadian&per=1960s,1970s,1980s,1990s,21st-century-and-contemporary&sort=newest",
    }



def count_classifications_from_xlsx(file_paths: list[Path]):
    # Load and combine data from all files
    dfs = [pd.read_excel(path) for path in file_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Drop rows without classification
    df = df.dropna(subset=['classification'])

    # Count occurrences
    counts = df['classification'].value_counts()

    # Print results
    print(f"\nFound {len(counts)} unique classifications:\n")
    for classification, count in counts.items():
        print(f"{classification}: {count} items")

    return counts


def combine_xlsx_files_to_fetch_all(file_paths: list[Path], drop_duplicates: bool = True):
    dfs = [pd.read_excel(path) for path in file_paths]

    for i, df in enumerate(dfs):
        print(f"File {i + 1} has {len(df)} rows")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined before deduplication: {len(combined_df)} rows")

    if drop_duplicates:
        combined_df = combined_df.drop_duplicates()
        print(f"Combined after deduplication: {len(combined_df)} rows")

    # Define output path: ../../data/fetch_ALL.xlsx
    output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fetch_ALL.xlsx"

    # Save
    combined_df.to_excel(output_path, index=False)
    print(f"Combined and saved to: {output_path}")
    print(f"Total rows: {len(combined_df)}")

    return combined_df


if __name__ == "__main__":
    # Use absolute paths from project root
    metadata_dir = project_root / "data" / "metadata"
    
    combine_xlsx_files_to_fetch_all([
        metadata_dir / "fetch_MoMA.xlsx",
        metadata_dir / "fetch_cooper_hewitt.xlsx", 
        metadata_dir / "mobile_phone_museum_data.xlsx",
        metadata_dir / "datamath_calculators.xlsx",
    ])


