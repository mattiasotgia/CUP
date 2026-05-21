"""
cup.core.filters
================
Built-in filter functions.  All are registered via ``@register_filter`` so
they become available automatically when the package is imported.

Adding a new filter
-------------------
1.  Write a function that accepts ``(df: pd.DataFrame, **params)`` and returns
    a filtered ``pd.DataFrame``.
2.  Decorate it with ``@register_filter("my_name")`` (optionally pass a
    ``describe`` callable for human-readable text on plots).
3.  Import this module (or your own module) before calling ``Config.load``.
"""

from typing import Optional, Tuple

import pandas as pd

from cup.core.registry import register_filter


# ---------------------------------------------------------------------------
# Describe helpers (kept separate for readability)
# ---------------------------------------------------------------------------

def _describe_filter_on(on: str, label: str,
                         min: Optional[float] = None,
                         max: Optional[float] = None) -> str:
    parts = []
    label_parts = label.split(":", 1)
    display_label = label_parts[0]
    unit = f" {label_parts[1]}" if len(label_parts) > 1 else ""

    if min is not None:
        parts.append(f"$\\geq {min:.2f}{unit}$")
    if max is not None:
        parts.append(f"$\\leq {max:.2f}{unit}$")

    suffix = "; ".join(parts)
    return f"{display_label} {suffix}" if suffix else display_label


# ---------------------------------------------------------------------------
# Built-in filters
# ---------------------------------------------------------------------------

@register_filter("max_slice_count")
def filter_max_slice_count(
    df: pd.DataFrame,
    event_column: str = "Evt",
    slice_column: str = "Slice",
    product: str = "sliceCount",
) -> pd.DataFrame:
    """
    For each event, compute slice count (= max slice ID + 1).
    Returns one row per event.
    """
    grouped = df.groupby(event_column)[slice_column].max()
    out = grouped.reset_index()
    out[product] = out[slice_column] + 1
    return out


@register_filter("filter_on", describe=_describe_filter_on)
def filter_filter_on(
    df: pd.DataFrame,
    on: str,
    label: str,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> pd.DataFrame:
    """Keep rows where ``df[on]`` is within [min, max] (both optional)."""
    mask = pd.Series(True, index=df.index)
    if min is not None:
        mask &= df[on] > min
    if max is not None:
        mask &= df[on] < max
    return df[mask]


@register_filter(
    "value_is",
    describe=lambda on, label, value=None: f"{label} = {value:.2f}" if value is not None else label,
)
def filter_value_is(
    df: pd.DataFrame,
    on: str,
    label: str,
    value: Optional[float] = None,
) -> pd.DataFrame:
    """Keep rows where ``df[on] == value``."""
    if value is None:
        return df
    return df[df[on] == value]


@register_filter("ratio", describe=lambda product, elements: None)
def filter_ratio(
    df: pd.DataFrame,
    product: str,
    elements: Tuple[str, str],
) -> pd.DataFrame:
    """Append a new column ``product = numerator / denominator``."""
    numerator, denominator = elements
    df = df.copy()
    df[product] = df[numerator] / df[denominator]
    return df

from cup.plotters._base_helpers import apply_filters
