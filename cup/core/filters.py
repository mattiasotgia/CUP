from cup.core.registry import register_filter
import pandas as pd

@register_filter("max_slice_count")
def filter_max_slice_count(
    df: pd.DataFrame,
    event_column="Evt",
    slice_column="Slice",
    product="sliceCount"
) -> pd.DataFrame:
    """
    For each event:
        - find max sliceId
        - convert to slice count = max sliceId + 1
    Returns a DataFrame with one row per event.
    """
    grouped = df.groupby(event_column)[slice_column].max()
    out = grouped.reset_index()
    out[product] = out[slice_column] + 1
    return out