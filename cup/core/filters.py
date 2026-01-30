'''
Docstring for cup.core.filters
'''

import pandas as pd
from typing import Tuple

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

def describe_filterOn(on, label, min = None, max = None):
    parts = []
    lu = label.split(":", 1)
    label = lu[0]
    unit = lu[1] if len(lu) > 1 else None
    unit = "" if not unit else f" {unit}"

    if min is not None:
        parts.append(f'$\\geq {min:.2f}~{unit}$')

    if max is not None:
        parts.append(f'$\\leq {max:.2f}~{unit}$')

    suffix = '; '.join(parts)
    text = f'{label} {suffix}' if suffix else label

    return text

@register_filter('filter_on', describe=describe_filterOn)   
def filter_filter_on(
    df: pd.DataFrame,
    on: str,
    label: str,
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


@register_filter(
    'value_is', 
    describe=lambda on, label, value = None: f'{label} = {value:.2f}'
)        
def filter_value_is(
    df: pd.DataFrame,
    on: str,
    label: str,
    value: float | None
): 
    '''
    Docstring for filter_value_is
    
    :param df: Dataframe of the data input
    :type df: pd.DataFrame
    :param on: Parameter over which filtering is done
    :type on: str
    :param label: Label of the parameter over which filtering is done
    :type label: str
    :param value: Paramenter value
    :type value: float | None
    '''

    mask = True

    if value:
        mask_value = df[on] == value
        mask = mask & mask_value

    return df[mask]

@register_filter('ratio', lambda product, elements: '')
def filter_ratio(
    df: pd.DataFrame,
    product: str,
    elements: Tuple[str, str]
):
    numerator, denominator = elements
    df[product] = df[numerator]/df[denominator]
    return df