"""Schema contract enforcement using YAML contract definitions.

Why data contracts:
  In senior production systems, schemas change unexpectedly. Upstream services
  might rename a column, change a data type, or start emitting nulls. If this
  garbage enters the pipeline, downstream ML training or inference fails silently.
  A data contract is a binding schema agreement between data producers and consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from shared.observability.logging import get_logger

log = get_logger(__name__)


class DataContractError(ValueError):
    """Raised when a DataFrame violates the defined data contract."""


def validate_contract(df: pd.DataFrame, contract_path: str | Path) -> None:
    """Validate a pandas DataFrame against a YAML schema contract.

    Args:
        df: The pandas DataFrame to validate.
        contract_path: Path to the YAML schema contract file.

    Raises:
        DataContractError: If any contract checks fail.
    """
    path = Path(contract_path)
    if not path.exists():
        raise FileNotFoundError(f"Data contract file not found: {path}")

    with open(path, "r") as f:
        contract = yaml.safe_load(f)

    schema_name = contract.get("name", "unknown")
    columns = contract.get("columns", {})

    errors = []

    # 1. Check for required columns
    for col_name, rules in columns.items():
        if rules.get("required", False) and col_name not in df.columns:
            errors.append(f"Required column '{col_name}' is missing.")

    # 2. Validate present columns
    for col_name in df.columns:
        if col_name not in columns:
            log.warning("unexpected_column_found", schema=schema_name, column=col_name)
            continue

        rules = columns[col_name]
        series = df[col_name]

        # Check data type
        expected_type = rules.get("type")
        if expected_type:
            # Map type strings to pandas/numpy types
            type_map = {
                "integer": "int",
                "float": "float",
                "string": "object",
                "boolean": "bool",
                "datetime": "datetime",
            }
            mapped_type = type_map.get(expected_type)
            if mapped_type:
                if mapped_type == "datetime":
                    if not pd.api.types.is_datetime64_any_dtype(series):
                        errors.append(
                            f"Column '{col_name}' type mismatch: expected datetime, got {series.dtype}"
                        )
                elif mapped_type == "int":
                    if not pd.api.types.is_integer_dtype(series):
                        errors.append(
                            f"Column '{col_name}' type mismatch: expected integer, got {series.dtype}"
                        )
                elif mapped_type == "float":
                    if not pd.api.types.is_float_dtype(series):
                        errors.append(
                            f"Column '{col_name}' type mismatch: expected float, got {series.dtype}"
                        )
                elif mapped_type == "bool":
                    if not pd.api.types.is_bool_dtype(series):
                        # Sometimes booleans are 0/1 integers
                        if not set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0, True, False}):
                            errors.append(
                                f"Column '{col_name}' type mismatch: expected boolean/binary, got {series.dtype}"
                            )
                elif mapped_type == "object":
                    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
                        errors.append(
                            f"Column '{col_name}' type mismatch: expected string, got {series.dtype}"
                        )

        # Check nullability
        allow_null = rules.get("allow_null", True)
        if not allow_null:
            null_count = int(series.isnull().sum())
            if null_count > 0:
                errors.append(f"Column '{col_name}' contains {null_count} null values (nulls not allowed).")

        # Check unique constraint
        unique = rules.get("unique", False)
        if unique:
            dupes = int(series.duplicated().sum())
            if dupes > 0:
                errors.append(f"Column '{col_name}' violated unique constraint: {dupes} duplicate values found.")

        # Check range rules
        val_min = rules.get("min")
        if val_min is not None:
            violators = series[series < val_min].dropna()
            if len(violators) > 0:
                errors.append(
                    f"Column '{col_name}' violated min threshold ({val_min}): "
                    f"found {len(violators)} values as low as {violators.min()}."
                )

        val_max = rules.get("max")
        if val_max is not None:
            violators = series[series > val_max].dropna()
            if len(violators) > 0:
                errors.append(
                    f"Column '{col_name}' violated max threshold ({val_max}): "
                    f"found {len(violators)} values as high as {violators.max()}."
                )

        # Check allowed values (enums)
        allowed_values = rules.get("allowed_values")
        if allowed_values:
            invalid = series[~series.isin(allowed_values)].dropna()
            if len(invalid) > 0:
                errors.append(
                    f"Column '{col_name}' contains values not in allowed list {allowed_values}: "
                    f"e.g., {invalid.unique()[:3]}."
                )

    if errors:
        log.error("data_contract_violation", schema=schema_name, error_count=len(errors))
        raise DataContractError(
            f"Data contract '{schema_name}' validation failed with {len(errors)} errors:\n" +
            "\n".join(f"- {err}" for err in errors)
        )

    log.info("data_contract_validated", schema=schema_name, columns=len(df.columns), rows=len(df))
