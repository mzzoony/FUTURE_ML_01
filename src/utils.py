"""
Utility functions for ML projects.

This module contains common helper functions used across different projects.
"""

import os
import json
import pickle
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        ValueError: If file format is not supported
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(filepath)
    elif ext == '.json':
        return pd.read_json(filepath)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path where model should be saved
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def create_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def save_results(results: Dict, filepath: str) -> None:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Dictionary containing results
        filepath: Path where results should be saved
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Print useful information about a dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print("=" * 50)
