"""
Data ingestion script for loading and validating procurement data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


# Expected columns for procurement dataset
REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "quantity",
    "unit_price",
    "supplier",
    "order_date",
    "delivery_date",
    "status",
]


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw procurement data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with raw data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records from {file_path}")
    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if all required columns exist
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ All {len(REQUIRED_COLUMNS)} required columns validated")
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for the dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }


def main(file_path: Optional[str] = None):
    """
    Main entry point for data loading and validation.
    """
    if file_path is None:
        # Default path to raw data
        file_path = "data/raw/procurement_data.csv"
    
    # Load data
    df = load_raw_data(file_path)
    
    # Validate columns
    validate_columns(df)
    
    # Get summary
    summary = get_data_summary(df)
    print(f"\nData Summary:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Duplicate rows: {summary['duplicates']}")
    
    return df


if __name__ == "__main__":
    main()