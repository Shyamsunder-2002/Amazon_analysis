"""
Streamlit Utilities - Handle dataframe serialization issues
"""

import pandas as pd
import streamlit as st

def prepare_df_for_streamlit(df):
    """
    Convert DataFrame columns to be compatible with Streamlit's Arrow serialization
    Fixes: "Could not convert dtype('O') with type numpy.dtypes.ObjectDType"
    """
    df_copy = df.copy()
    
    # Convert object columns to string
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('string')
    
    # Convert datetime columns to string for display
    for col in df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Handle period objects
    for col in df_copy.columns:
        if hasattr(df_copy[col], 'dt') and hasattr(df_copy[col].dt, 'to_timestamp'):
            df_copy[col] = df_copy[col].dt.to_timestamp().astype('string')
    
    return df_copy

def safe_column_access(df, column_name, default_value=None):
    """
    Safely access DataFrame column, return default if column doesn't exist
    """
    if column_name in df.columns:
        return df[column_name]
    else:
        st.warning(f"⚠️ Column '{column_name}' not found. Using default value.")
        if default_value is not None:
            return pd.Series([default_value] * len(df), index=df.index)
        else:
            return pd.Series([0] * len(df), index=df.index)

def display_dataframe_safe(df, title="Data Preview"):
    """
    Safely display DataFrame with proper serialization
    """
    try:
        df_display = prepare_df_for_streamlit(df)
        st.subheader(title)
        st.dataframe(df_display, width="stretch")
    except Exception as e:
        st.error(f"Error displaying dataframe: {e}")
        st.write("Dataframe shape:", df.shape)
        st.write("Columns:", list(df.columns))




