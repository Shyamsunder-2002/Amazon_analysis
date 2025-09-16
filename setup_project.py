"""
Complete setup script with ALL required columns for Amazon India Analytics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_project_structure():
    """Create complete project structure"""
    base_dir = Path.cwd()
    
    directories = [
        'data/raw', 'data/cleaned', 'data/processed',
        'src', 'config', 'pages', 'sql', '.streamlit'
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def generate_complete_dataset():
    """Generate dataset with ALL required columns"""
    
    print("🔄 Generating COMPLETE Amazon India dataset...")
    
    np.random.seed(42)
    n_records = 50000
    
    # Base data
    categories = ['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors', 'Books', 'Beauty & Personal Care']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune']
    brands = ['Samsung', 'Apple', 'Nike', 'Adidas', 'LG', 'Sony', 'Puma', 'Reebok']
    payment_methods = ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery']
    shipping_partners = ['Ekart', 'BlueDart', 'Delhivery', 'FedEx', 'IndiaPost']
    warehouses = ['Mumbai_WH', 'Delhi_WH', 'Bangalore_WH', 'Chennai_WH']
    
    # Complete dataset with ALL columns
    data = {
        # Basic identifiers
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
        'customer_id': [f'CUST_{np.random.randint(1, 15000):05d}' for _ in range(n_records)],
        'product_id': [f'PROD_{np.random.randint(1, 3000):04d}' for _ in range(n_records)],
        
        # Date information
        'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
        
        # Product information
        'category': np.random.choice(categories, n_records),
        'subcategory': [f'Sub_{np.random.randint(1, 20)}' for _ in range(n_records)],
        'brand': np.random.choice(brands, n_records),
        'product_name': [f'Product_{np.random.randint(1, 1000)}' for _ in range(n_records)],
        
        # Financial data
        'original_price_inr': np.random.gamma(2, 1500, n_records),
        'discount_percent': np.random.uniform(0, 60, n_records),
        'final_amount_inr': np.random.gamma(2, 1200, n_records),
        'delivery_charges': np.random.uniform(0, 300, n_records),
        
        # Customer data
        'customer_city': np.random.choice(cities, n_records),
        'customer_state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'West Bengal'], n_records),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_records),
        'is_prime_member': np.random.choice([True, False], n_records, p=[0.35, 0.65]),
        
        # Order details
        'payment_method': np.random.choice(payment_methods, n_records),
        'delivery_days': np.random.choice(range(0, 8), n_records),
        'customer_rating': np.random.uniform(1, 5, n_records),
        'product_rating': np.random.uniform(1, 5, n_records),
        'is_festival_sale': np.random.choice([True, False], n_records, p=[0.20, 0.80]),
        'festival_name': np.random.choice(['Diwali', 'Prime Day', 'Great Indian Sale', None], n_records, p=[0.08, 0.06, 0.06, 0.80]),
        'return_status': np.random.choice(['Not Returned', 'Returned', 'Exchanged'], n_records, p=[0.85, 0.12, 0.03]),
        
        # Operations data
        'shipping_partner': np.random.choice(shipping_partners, n_records),
        'warehouse_location': np.random.choice(warehouses, n_records),
        'order_status': np.random.choice(['Delivered', 'In Transit', 'Processing', 'Cancelled'], n_records, p=[0.85, 0.05, 0.05, 0.05]),
        'is_prime_eligible': np.random.choice([True, False], n_records, p=[0.70, 0.30]),
        
        # Advanced analytics columns
        'customer_tenure_days': np.random.randint(1, 3650, n_records),
        'previous_orders': np.random.randint(0, 100, n_records),
        'avg_order_value_history': np.random.uniform(500, 5000, n_records),
        'will_churn': np.random.choice([True, False], n_records, p=[0.15, 0.85]),
        'will_return_next_month': np.random.choice([True, False], n_records, p=[0.35, 0.65]),
        'lifetime_value': np.random.uniform(1000, 50000, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add ALL calculated columns
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_quarter'] = df['order_date'].dt.quarter
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = df['order_day_of_week'].isin([5, 6])  # MISSING COLUMN FIXED
    df['discount_amount'] = df['original_price_inr'] * df['discount_percent'] / 100
    df['gross_revenue'] = df['final_amount_inr'] + df['delivery_charges']  # MISSING COLUMN FIXED
    df['on_time_delivery'] = df['delivery_days'] <= 3  # MISSING COLUMN FIXED
    df['delivery_date'] = df['order_date'] + pd.to_timedelta(df['delivery_days'], unit='D')
    df['delivery_delay'] = np.where(df['delivery_days'] > 3, df['delivery_days'] - 3, 0)
    df['order_processing_time'] = np.random.uniform(0.5, 2.0, n_records)
    df['distance_km'] = np.random.uniform(50, 2000, n_records)
    df['delivery_cost'] = df['delivery_charges'] + np.random.uniform(50, 200, n_records)
    df['customer_value_segment'] = pd.cut(df['avg_order_value_history'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Save dataset
    data_dir = Path('data/raw')
    df.to_csv(data_dir / 'amazon_india_complete_2015_2025.csv', index=False)
    
    print(f"✅ Generated dataset: {len(df):,} transactions")
    print(f"✅ Total columns: {len(df.columns)}")
    print(f"✅ Missing column issues should be resolved!")

def create_utils_file():
    """Create src/utils.py with helper functions"""
    
    utils_content = '''"""
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
'''
    
    src_dir = Path('src')
    with open(src_dir / 'utils.py', 'w') as f:
        f.write(utils_content)
    
    print("✅ Created src/utils.py with helper functions")

def create_streamlit_config():
    """Create fixed Streamlit configuration"""
    
    config_dir = Path('.streamlit')
    
    config_content = """[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = true
enableXsrfProtection = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[runner]
magicEnabled = true

[browser]
gatherUsageStats = false
"""
    
    with open(config_dir / 'config.toml', 'w') as f:
        f.write(config_content)
    
    print("✅ Created fixed Streamlit configuration")

if __name__ == "__main__":
    print("🚀 Complete Amazon India Analytics Setup...")
    print("="*60)
    
    create_project_structure()
    generate_complete_dataset()
    create_utils_file()
    create_streamlit_config()
    
    print("\n🎉 ALL FIXES APPLIED!")
    print("\n📋 Next steps:")
    print("1. Run: streamlit run Home.py")
    print("2. All errors should be resolved!")




