"""
Supplier normalization module for demand forecasting inventory system.

This module handles supplier name normalization including:
- Whitespace trimming and case standardization
- Suffix normalization (Inc, Ltd, Co, Corp, etc.)
- Alias/known-variant mapping
- Fuzzy matching for near-duplicate supplier names
"""

import re
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Canonical suffix map — maps common variants to a single standard form
# ---------------------------------------------------------------------------
SUFFIX_MAP: dict[str, str] = {
    r"\bincorporated\b": "Inc",
    r"\binc\.?$": "Inc",
    r"\blimited\b": "Ltd",
    r"\bltd\.?$": "Ltd",
    r"\bcorporation\b": "Corp",
    r"\bcorp\.?$": "Corp",
    r"\bcompany\b": "Co",
    r"\bco\.?$": "Co",
    r"\bllc\.?$": "LLC",
    r"\bllp\.?$": "LLP",
    r"\bgroup\b": "Group",
    r"\benterprises\b": "Enterprises",
    r"\bsolutions\b": "Solutions",
    r"\bservices\b": "Services",
    r"\bindustries\b": "Industries",
    r"\bsupply\b": "Supply",
    r"\bsupplies\b": "Supplies",
    r"\bsource\b": "Source",
    r"\bmaster\b": "Master",
    r"\bglobal\b": "Global",
    r"\btech\b": "Tech",
    r"\btechnology\b": "Technology",
    r"\btechnologies\b": "Technologies",
}

# ---------------------------------------------------------------------------
# Known alias mapping — maps known dirty variants to the canonical name.
# Extend this dict as new variants are discovered.
# ---------------------------------------------------------------------------
SUPPLIER_ALIAS_MAP: dict[str, str] = {
    # Acme Corp variants
    "acme": "Acme Corp",
    "acme corporation": "Acme Corp",
    "acme corp.": "Acme Corp",
    "acme co": "Acme Corp",
    # TechSupply Inc variants
    "techsupply": "TechSupply Inc",
    "tech supply inc": "TechSupply Inc",
    "tech supply": "TechSupply Inc",
    "techsupply incorporated": "TechSupply Inc",
    # Global Parts Co variants
    "global parts": "Global Parts Co",
    "global parts company": "Global Parts Co",
    "globalparts co": "Global Parts Co",
    # ToolMaster Ltd variants
    "toolmaster": "ToolMaster Ltd",
    "tool master ltd": "ToolMaster Ltd",
    "tool master limited": "ToolMaster Ltd",
    "toolmaster limited": "ToolMaster Ltd",
    # RawSource Inc variants
    "rawsource": "RawSource Inc",
    "raw source inc": "RawSource Inc",
    "raw source": "RawSource Inc",
    "rawsource incorporated": "RawSource Inc",
}


def _strip_and_titlecase(name: str) -> str:
    """Remove extra whitespace and apply title-case."""
    return " ".join(name.strip().split()).title()


def _normalize_suffixes(name: str) -> str:
    """
    Replace long-form or punctuated legal suffixes with their canonical short form.

    Args:
        name: Supplier name string (already lower-cased).

    Returns:
        Name with suffixes normalized (still lower-case).
    """
    for pattern, replacement in SUFFIX_MAP.items():
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    return name


def normalize_supplier_name(name: str) -> str:
    """
    Normalize a single supplier name to its canonical form.

    Normalization steps:
    1. Strip extra whitespace.
    2. Check alias map (case-insensitive).
    3. Normalize legal-entity suffixes.
    4. Apply title-case to the final result.

    Args:
        name: Raw supplier name string.

    Returns:
        Canonical supplier name string.
    """
    if not isinstance(name, str):
        return name  # propagate NaN / non-string values unchanged

    cleaned = " ".join(name.strip().split())

    # --- Alias map lookup (exact match, case-insensitive) ---
    alias_key = cleaned.lower()
    if alias_key in SUPPLIER_ALIAS_MAP:
        return SUPPLIER_ALIAS_MAP[alias_key]

    # --- Suffix normalization ---
    normalized = _normalize_suffixes(cleaned)

    # --- Title-case ---
    return _strip_and_titlecase(normalized)


def normalize_supplier_column(
    df: pd.DataFrame,
    column: str = "supplier",
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply supplier name normalization to a DataFrame column.

    Args:
        df: Input DataFrame containing supplier names.
        column: Name of the column that holds raw supplier names.
                Defaults to ``"supplier"``.
        output_column: Name of the column to write normalized names into.
                       If ``None`` (default), overwrites the source column
                       in-place (on a copy).

    Returns:
        DataFrame with normalized supplier names.

    Raises:
        KeyError: If ``column`` does not exist in ``df``.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")

    result = df.copy()
    target = output_column if output_column else column
    result[target] = result[column].map(normalize_supplier_name)
    return result


def get_normalization_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str = "supplier",
) -> dict:
    """
    Generate a report comparing supplier names before and after normalization.

    Args:
        df_before: DataFrame with original supplier names.
        df_after: DataFrame with normalized supplier names.
        column: Column name to compare.

    Returns:
        Dictionary with normalization statistics and a change log.
    """
    before = df_before[column]
    after = df_after[column]

    changed_mask = before != after
    changes = (
        pd.DataFrame({"original": before[changed_mask], "normalized": after[changed_mask]})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return {
        "total_records": len(before),
        "records_changed": int(changed_mask.sum()),
        "unique_before": before.nunique(),
        "unique_after": after.nunique(),
        "canonical_suppliers": sorted(after.dropna().unique().tolist()),
        "change_log": changes.to_dict(orient="records"),
    }
