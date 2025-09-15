"""
Complete Streamlit utilities for Amazon India Analytics Platform
ALL FIXES INCLUDED
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def safe_column_access(df, col_name, default_value=0):
    """Safely access DataFrame column with fallback"""
    if col_name in df.columns:
        return df[col_name]
    else:
        if callable(default_value):
            try:
                return default_value()
            except:
                return pd.Series([0] * len(df), index=df.index)
        elif isinstance(default_value, (list, tuple, np.ndarray)):
            return pd.Series(default_value, index=df.index)
        else:
            return pd.Series([default_value] * len(df), index=df.index)

def fix_dataframe_for_streamlit(df):
    """Fix ALL DataFrame serialization issues for Streamlit/Arrow"""
    if df is None or df.empty:
        return df
        
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Convert object columns to string
        if pd.api.types.is_object_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype('string').fillna("")
        
        # Convert boolean to int
        elif pd.api.types.is_bool_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(int)
        
        # Fix datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d').fillna("")
        
        # Ensure numeric columns are clean
        elif pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    
    return df_copy

def safe_dataframe_display(df, **kwargs):
    """Display DataFrame with fixed parameters and serialization"""
    # Fix deprecated parameter
    if 'use_container_width' in kwargs:
        use_container_width = kwargs.pop('use_container_width')
        kwargs['width'] = "stretch" if use_container_width else "content"
    
    # Fix serialization
    df_fixed = fix_dataframe_for_streamlit(df)
    return st.dataframe(df_fixed, **kwargs)

def safe_plotly_chart(fig, **kwargs):
    """Display Plotly chart with fixed parameters"""
    if 'use_container_width' in kwargs:
        kwargs.pop('use_container_width')
    return st.plotly_chart(fig, **kwargs)

def safe_metric_calculation(series, operation='mean'):
    """Safely calculate metrics from series"""
    if series is None or series.empty:
        return 0.0
    
    try:
        if operation == 'mean':
            return float(series.mean())
        elif operation == 'sum':
            return float(series.sum())
        elif operation == 'count':
            return int(len(series))
        elif operation == 'rate':
            return float((series.sum() / len(series)) * 100)
        else:
            return 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0

def prepare_ml_features_safe(df):
    """Prepare ML features with complete safety checks"""
    features_df = df.copy()
    
    # Ensure all required columns exist
    required_cols = {
        'order_date': pd.Timestamp.now(),
        'final_amount_inr': 1000,
        'discount_percent': 10,
        'customer_rating': 4.0,
        'delivery_days': 3,
        'customer_tenure_days': 365,
        'previous_orders': 1,
        'avg_order_value_history': 1000,
        'category': 'Electronics',
        'customer_city': 'Mumbai',
        'age_group': '26-35',
        'payment_method': 'UPI',
        'is_prime_member': False
    }
    
    for col, default_val in required_cols.items():
        if col not in features_df.columns:
            features_df[col] = default_val
    
    # Create derived columns safely
    if 'order_date' in features_df.columns:
        features_df['order_year'] = pd.to_datetime(features_df['order_date']).dt.year
        features_df['order_month'] = pd.to_datetime(features_df['order_date']).dt.month
        features_df['order_day_of_week'] = pd.to_datetime(features_df['order_date']).dt.dayofweek
    else:
        features_df['order_year'] = 2024
        features_df['order_month'] = 6
        features_df['order_day_of_week'] = 1
    
    features_df['is_weekend'] = features_df['order_day_of_week'].isin([5, 6]).astype(int)
    
    # Encode categorical variables safely
    categorical_cols = ['category', 'customer_city', 'age_group', 'payment_method']
    
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
        except:
            features_df[f'{col}_encoded'] = 0
    
    # Convert boolean to int
    features_df['is_prime_member'] = features_df['is_prime_member'].astype(int)
    
    # Define ML features list
    ml_features = [
        'final_amount_inr', 'discount_percent', 'customer_rating', 'delivery_days',
        'customer_tenure_days', 'previous_orders', 'avg_order_value_history',
        'category_encoded', 'customer_city_encoded', 'age_group_encoded', 'payment_method_encoded',
        'is_prime_member', 'order_month', 'order_day_of_week', 'is_weekend'
    ]
    
    # Ensure all ML features exist
    for feature in ml_features:
        if feature not in features_df.columns:
            features_df[feature] = 0
    
    return features_df, ml_features
