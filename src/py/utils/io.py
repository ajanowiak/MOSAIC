# io.py
"""Shared I/O helpers for MOSAIC pipeline scripts."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with the first column as the row index.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame indexed by the first column.
    """
    return pd.read_csv(path, index_col=0)


def save_csv(df: pd.DataFrame, path: str, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating parent directories as needed.

    Args:
        df: DataFrame to write.
        path: Output CSV path.
        index: Whether to write the row index.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    logger.info("Saved %s", output_path)
