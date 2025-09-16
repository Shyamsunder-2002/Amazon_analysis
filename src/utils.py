"""
Universal utility functions for Amazon India Analytics Platform
"""

import pandas as pd
import numpy as np
import streamlit as st

def safe_column_access(df, col_name, default_value=0):
    """Safely access DataFrame column, return default if column doesn't exist"""
    if col_name in df.columns:
        return df[col_name]
    else:
        if isinstance(default_value, (int, float)):
            return pd.Series([default_value] * len(df), index=df.index)
        elif callable(default_value):
            return default_value(df)
        else:
            return pd.Series([default_value] * len(df), index=df.index)

def prepare_df_for_streamlit(df):
    """Convert DataFrame to be Arrow-compatible"""
    df_copy = df.copy()
    
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('string')
    
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(df_copy[col].dtype, pd.PeriodDtype):
            df_copy[col] = df_copy[col].dt.to_timestamp().dt.strftime('%Y-%m-%d')
    
    return df_copy

def st.dataframe(df, **kwargs):
    """Handle deprecated use_container_width parameter"""
    if 'use_container_width' in kwargs:
        use_container_width = kwargs.pop('use_container_width')
        kwargs['width'] = "stretch" if use_container_width else "content"
    
    df_prepared = prepare_df_for_streamlit(df)
    return st.dataframe(df_prepared, **kwargs)
