"""
Product Analytics Page - Product Performance & Inventory Analysis
Deep dive into product metrics, catalog performance, and inventory insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

from pathlib import Path

# Define data directory
CLEANED_DATA_DIR = Path("data/cleaned")

# Safe imports with fallbacks
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils import safe_column_access, fix_dataframe_for_streamlit, safe_dataframe_display, safe_plotly_chart
except ImportError:
    def safe_column_access(df, col_name, default_value=0):
        if col_name in df.columns:
            return df[col_name]
        else:
            return pd.Series([default_value] * len(df), index=df.index)
    
    def fix_dataframe_for_streamlit(df):
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_object_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype('string').fillna("")
        return df_copy
    
    def safe_dataframe_display(df, **kwargs):
        """Fallback safe dataframe display function"""
        
        # Handle width parameter correctly
        width = kwargs.pop('width', None)
        
        if width == 'stretch':
            kwargs['use_container_width'] = True
        elif width == 'content':
            kwargs['width'] = 700
        elif isinstance(width, int):
            kwargs['width'] = width
        
        # Handle Series input (convert to DataFrame)
        if hasattr(df, 'to_frame'):  # It's a Series
            df = df.to_frame()
        
        # Fix DataFrame serialization
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_object_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype('string').fillna("")
            elif pd.api.types.is_bool_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(int)
        
        # CRITICAL: Call st.dataframe, NOT safe_dataframe_display!
        return st.dataframe(df_copy, **kwargs)

    
    def safe_plotly_chart(fig, **kwargs):
        if 'use_container_width' in kwargs:
            kwargs.pop('use_container_width')
        return st.plotly_chart(fig, **kwargs)

st.set_page_config(
    page_title="📦 Product Analytics",
    page_icon="📦",
    layout="wide"
)

@st.cache_data
def load_product_data():
    """Load data optimized for product analysis"""
    try:
        data_path = 'data/raw/amazon_india_complete_2015_2025.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            # Generate product-focused dataset
            np.random.seed(42)
            n_records = 150000
            
            categories = ['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors', 'Books', 'Beauty & Personal Care']
            brands = ['Samsung', 'Apple', 'Nike', 'Adidas', 'LG', 'Sony', 'Puma', 'Reebok']
            
            data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
                'customer_id': [f'CUST_{np.random.randint(1, 40000):05d}' for _ in range(n_records)],
                'product_id': [f'PROD_{np.random.randint(1, 8000):04d}' for _ in range(n_records)],
                'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
                'category': np.random.choice(categories, n_records),
                'brand': np.random.choice(brands, n_records),
                'final_amount_inr': np.random.gamma(2, 1800, n_records),
                'product_rating': np.random.uniform(1, 5, n_records),
                'customer_rating': np.random.uniform(1, 5, n_records),
                'discount_percent': np.random.uniform(0, 80, n_records),
                'return_status': np.random.choice(['Not Returned', 'Returned', 'Exchanged'], n_records, p=[0.88, 0.10, 0.02]),
                'is_prime_eligible': np.random.choice([True, False], n_records, p=[0.75, 0.25]),
                'customer_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'], n_records)
            }
            
            df = pd.DataFrame(data)
            df['product_name'] = df['category'] + '_' + df['brand']
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month
            return df
    except Exception as e:
        st.error(f"Error loading product data: {e}")
        return None

def calculate_product_metrics(df):
    """Calculate comprehensive product metrics"""
    
    # Product performance metrics
    product_summary = df.groupby('product_id').agg({
        'final_amount_inr': ['sum', 'mean', 'count'],
        'product_rating': 'mean',
        'customer_rating': 'mean',
        'discount_percent': 'mean',
        'return_status': lambda x: (x == 'Returned').sum()
    }).round(2)
    
    product_summary.columns = ['total_revenue', 'avg_price', 'total_sales', 'avg_product_rating', 'avg_customer_rating', 'avg_discount', 'returns']
    product_summary['return_rate'] = (product_summary['returns'] / product_summary['total_sales']) * 100
    
    # Category summary
    category_summary = df.groupby('category').agg({
        'final_amount_inr': ['sum', 'mean'],
        'product_id': 'nunique',
        'transaction_id': 'count',
        'product_rating': 'mean',
        'return_status': lambda x: (x == 'Returned').sum()
    }).round(2)
    
    category_summary.columns = ['total_revenue', 'avg_order_value', 'unique_products', 'total_orders', 'avg_rating', 'returns']
    category_summary['return_rate'] = (category_summary['returns'] / category_summary['total_orders']) * 100
    
    # Brand summary
    brand_summary = df.groupby('brand').agg({
        'final_amount_inr': ['sum', 'mean'],
        'product_rating': 'mean',
        'transaction_id': 'count'
    }).round(2)
    
    brand_summary.columns = ['total_revenue', 'avg_price', 'avg_rating', 'total_sales']
    
    # Main metrics
    metrics = {
        'total_products': df['product_id'].nunique(),
        'total_categories': df['category'].nunique(),
        'total_brands': df['brand'].nunique(),
        'avg_product_rating': df['product_rating'].mean(),
        'total_product_revenue': df['final_amount_inr'].sum(),
        'avg_product_price': df['final_amount_inr'].mean(),
        'overall_return_rate': (df['return_status'] == 'Returned').sum() / len(df) * 100,
        'prime_eligible_products': (safe_column_access(df, 'is_prime_eligible', False).sum() / len(df)) * 100
    }
    
    return metrics, product_summary, category_summary, brand_summary

def display_product_kpis(metrics):
    """Display product KPI cards"""
    st.markdown("## 📦 Product Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total Products", f"{metrics['total_products']:,}")
    
    with col2:
        st.metric("⭐ Avg Rating", f"{metrics['avg_product_rating']:.2f}/5.0")
    
    with col3:
        st.metric("💰 Avg Price", f"₹{metrics['avg_product_price']:,.0f}")
    
    with col4:
        st.metric("↩️ Return Rate", f"{metrics['overall_return_rate']:.1f}%")

def create_category_brand_analysis(df, category_summary, brand_summary):
    """Create category and brand analysis - FIXED VERSION"""
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=('Revenue Distribution by Category', 'Top 10 Cities by Revenue',
                       'Revenue by Payment Method', 'Brand Performance')
    )
    
    # Category revenue pie chart
    category_revenue = category_summary['total_revenue'].sort_values(ascending=False)
    fig.add_trace(
        go.Pie(labels=category_revenue.index, values=category_revenue.values,
               name="Category Revenue"),
        row=1, col=1
    )
    
    # City breakdown
    city_revenue = df.groupby('customer_city')['final_amount_inr'].sum().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(x=city_revenue.index, y=city_revenue.values/1e6,
               name="City Revenue", marker_color='#FFD700'),
        row=1, col=2
    )
    
    # Payment method breakdown
    if 'payment_method' in df.columns:
        payment_revenue = df.groupby('payment_method')['final_amount_inr'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=payment_revenue.index, y=payment_revenue.values/1e6,
                   name="Payment Revenue", marker_color='#FFA500'),
            row=2, col=1
        )
    
    # Brand performance
    top_brands = brand_summary.nlargest(12, 'total_revenue')
    fig.add_trace(
        go.Bar(x=top_brands.index, y=top_brands['avg_rating'],
               name="Brand Rating", marker_color='#4ECDC4'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="🎯 Category & Brand Analysis",
        height=700,
        showlegend=False
    )
    
    return fig

def main():
    """Main product analytics page"""
    
    st.markdown('<h1 style="text-align: center; color: #4ECDC4;">📦 Product Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading product data..."):
        df = load_product_data()
    
    if df is None:
        st.error("❌ Failed to load product data")
        return
    
    # Calculate metrics
    metrics, product_summary, category_summary, brand_summary = calculate_product_metrics(df)
    
    # Display KPIs
    display_product_kpis(metrics)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Product Filters")
        
        selected_categories = st.multiselect(
            "🏷️ Categories:",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        
        selected_brands = st.multiselect(
            "🏢 Brands:",
            options=df['brand'].unique(),
            default=df['brand'].unique()[:5]
        )
        
        rating_range = st.slider(
            "⭐ Product Rating Range:",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.1
        )
    
    # Filter data
    df_filtered = df[
        (df['category'].isin(selected_categories)) &
        (df['brand'].isin(selected_brands)) &
        (df['product_rating'] >= rating_range[0]) &
        (df['product_rating'] <= rating_range[1])
    ]
    
    if df_filtered.empty:
        st.warning("⚠️ No products match the selected filters")
        return
    
    # Recalculate metrics for filtered data
    filtered_metrics, filtered_product_summary, filtered_category_summary, filtered_brand_summary = calculate_product_metrics(df_filtered)
    
    # Main analytics
    st.markdown("## 📊 Product Performance Analysis")
    performance_chart = create_category_brand_analysis(df_filtered, filtered_category_summary, filtered_brand_summary)
    safe_plotly_chart(performance_chart)
    
    # Top products table
    st.markdown("## 🏆 Top Performing Products")
    
    top_products = filtered_product_summary.nlargest(20, 'total_revenue')[['total_revenue', 'total_sales', 'avg_product_rating', 'return_rate']]
    st.dataframe(top_products, use_container_width=True)
    
    # Category comparison
    st.markdown("## 📈 Category Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cat_rev = px.bar(
            x=filtered_category_summary.index,
            y=filtered_category_summary['total_revenue']/1e6,
            title="Category Revenue (₹M)"
        )
        safe_plotly_chart(fig_cat_rev)
    
    with col2:
        fig_cat_rating = px.bar(
            x=filtered_category_summary.index,
            y=filtered_category_summary['avg_rating'],
            title="Category Average Rating"
        )
        safe_plotly_chart(fig_cat_rating)

if __name__ == "__main__":
    main()




