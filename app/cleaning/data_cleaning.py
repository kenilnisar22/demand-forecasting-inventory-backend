"""
Data cleaning module for demand forecasting inventory system.

This module handles data cleaning operations including:
- Duplicate removal
- Missing value handling
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class DataCleaner:
    """Handles data cleaning operations for the inventory forecasting system."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Args:
            df: Input pandas DataFrame to clean
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_report = {}
    
    def remove_duplicates(self, subset: Optional[list] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset: List of columns to consider for identifying duplicates.
                   If None, all columns are considered.
            keep: Which duplicates to keep ('first', 'last', or False to remove all).
                 Defaults to 'first'.
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = initial_rows - len(self.df)
        
        self.cleaning_report['duplicates_removed'] = removed_rows
        
        return self.df
    
    def handle_missing_values(
        self,
        strategy: str = 'drop',
        threshold: float = 0.5,
        numeric_method: str = 'mean',
        categorical_method: str = 'mode'
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            strategy: Strategy to handle missing values:
                     - 'drop': Drop rows with missing values
                     - 'fill': Fill missing values based on numeric/categorical methods
                     - 'hybrid': Drop rows where missing percentage > threshold, 
                               fill others
            threshold: Percentage threshold (0-1) for dropping rows when using 'hybrid' strategy
            numeric_method: Method to fill numeric columns ('mean', 'median', 'ffill', 'bfill')
            categorical_method: Method to fill categorical columns ('mode', 'ffill', 'bfill')
        
        Returns:
            DataFrame with missing values handled
        """
        initial_missing = self.df.isnull().sum().sum()
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        
        elif strategy == 'fill':
            self.df = self._fill_missing_values(numeric_method, categorical_method)
        
        elif strategy == 'hybrid':
            # Drop columns with missing percentage above threshold
            missing_pct = self.df.isnull().sum() / len(self.df)
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            
            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
            
            # Fill remaining missing values
            self.df = self._fill_missing_values(numeric_method, categorical_method)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'fill', or 'hybrid'.")
        
        final_missing = self.df.isnull().sum().sum()
        
        self.cleaning_report['missing_values_handled'] = {
            'initial_count': initial_missing,
            'final_count': final_missing,
            'strategy': strategy
        }
        
        return self.df
    
    def _fill_missing_values(self, numeric_method: str, categorical_method: str) -> pd.DataFrame:
        """
        Fill missing values based on column data types.
        
        Args:
            numeric_method: Method for numeric columns
            categorical_method: Method for categorical columns
        
        Returns:
            DataFrame with filled missing values
        """
        df = self.df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns
        for col in numeric_cols:
            if df[col].isnull().any():
                if numeric_method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif numeric_method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif numeric_method == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                elif numeric_method == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                    df[col].fillna(method='ffill', inplace=True)
        
        # Fill categorical columns
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_method == 'mode':
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                elif categorical_method == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
                elif categorical_method == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                    df[col].fillna(method='ffill', inplace=True)
        
        return df
    
    def get_cleaning_report(self) -> dict:
        """
        Get a report of all cleaning operations performed.
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'operations': self.cleaning_report
        }
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            Cleaned DataFrame
        """
        return self.df


def clean_data(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: bool = True,
    missing_strategy: str = 'hybrid',
    duplicate_subset: Optional[list] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform complete data cleaning on a DataFrame.
    
    Args:
        df: Input DataFrame to clean
        remove_duplicates: Whether to remove duplicates
        handle_missing: Whether to handle missing values
        missing_strategy: Strategy for handling missing values
        duplicate_subset: Columns to consider for duplicate detection
    
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    cleaner = DataCleaner(df)
    
    if remove_duplicates:
        cleaner.remove_duplicates(subset=duplicate_subset)
    
    if handle_missing:
        cleaner.handle_missing_values(strategy=missing_strategy)
    
    return cleaner.get_cleaned_data(), cleaner.get_cleaning_report()
