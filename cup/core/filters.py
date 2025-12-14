'''
Docstring for cup.core.filters
'''

import pandas as pd

from cup.core.registry import register_filter

@register_filter('max_slice_count')
def filter_max_slice_count(
    df: pd.DataFrame,
    event_column: str = 'Evt',
    slice_column: str = 'Slice',
    product: str = 'sliceCount'
) -> pd.DataFrame:
    '''
    Docstring for filter_max_slice_count
    
    :param df: Dataframe of the data input
    :type df: pd.DataFrame
    :param event_column: Event column
    :type event_column: str
    :param slice_column: Slice column
    :type slice_column: str
    :param product: what is the out product called
    :type product: str
    :return: Filtered dataframe
    
    For each event:
        - find max sliceId
        - convert to slice count = max sliceId + 1
    Returns a DataFrame with one row per event.
    :rtype: DataFrame
    
    
    '''
    grouped = df.groupby(event_column)[slice_column].max()
    out = grouped.reset_index()
    out[product] = out[slice_column] + 1
    return out

@register_filter('filter_on')
def filter_filter_on(
    df: pd.DataFrame,
    on: str,
    min: float | None = None,
    max: float | None = None 
): 
    '''
    Docstring for filter_filter_on
    
    :param df: Dataframe of the data input
    :type df: pd.DataFrame
    :param on: Value over which filtering is done
    :type on: str
    :param min: Min, for value
    :type min: float | None
    :param max: Max, for value
    :type max: float | None

    :return: Filtered dataframe
    :rtype: DataFrame

    For each event (row) filter based on the values of the 'on' variable
    '''

    mask = True

    if min:
        mask_min = df[on] > min
        mask = mask & mask_min
    if max:
        mask_max = df[on] < max
        mask = mask & mask_max

    return df[mask]