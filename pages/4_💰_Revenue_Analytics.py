"""
Revenue Analytics Page - Financial Performance Analysis
Deep dive into revenue metrics, trends, and financial KPIs
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
    page_title="💰 Revenue Analytics",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def load_revenue_data():
    """Load data optimized for revenue analysis"""
    try:
        data_path = 'data/raw/amazon_india_complete_2015_2025.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            # Generate revenue-focused dataset
            np.random.seed(42)
            n_records = 100000
            
            data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
                'customer_id': [f'CUST_{np.random.randint(1, 25000):05d}' for _ in range(n_records)],
                'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
                'category': np.random.choice(['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors', 'Books', 'Beauty & Personal Care'], n_records),
                'final_amount_inr': np.random.gamma(2, 1200, n_records),
                'discount_percent': np.random.uniform(0, 70, n_records),
                'customer_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune'], n_records),
                'is_prime_member': np.random.choice([True, False], n_records, p=[0.42, 0.58]),
                'is_festival_sale': np.random.choice([True, False], n_records, p=[0.28, 0.72]),
                'payment_method': np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery'], n_records),
                'delivery_charges': np.random.uniform(0, 200, n_records)
            }
            
            df = pd.DataFrame(data)
            df['original_price_inr'] = df['final_amount_inr'] / (1 - df['discount_percent']/100)
            df['discount_amount'] = df['original_price_inr'] - df['final_amount_inr']
            df['gross_revenue'] = df['final_amount_inr'] + df['delivery_charges']
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month
            df['order_quarter'] = df['order_date'].dt.quarter
            return df
    except Exception as e:
        st.error(f"Error loading revenue data: {e}")
        return None

def calculate_revenue_metrics(df):
    """Calculate comprehensive revenue metrics"""
    current_year = df['order_year'].max()
    previous_year = current_year - 1
    
    current_data = df[df['order_year'] == current_year]
    previous_data = df[df['order_year'] == previous_year]
    
    current_revenue = current_data['final_amount_inr'].sum()
    previous_revenue = previous_data['final_amount_inr'].sum()
    
    metrics = {
        'total_revenue': df['final_amount_inr'].sum(),
        'current_year_revenue': current_revenue,
        'previous_year_revenue': previous_revenue,
        'gross_revenue': safe_column_access(df, 'gross_revenue', df['final_amount_inr']).sum(),
        'total_discount_given': safe_column_access(df, 'discount_amount', pd.Series([0] * len(df))).sum(),
        'revenue_growth': ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0,
        'average_order_value': df['final_amount_inr'].mean(),
        'revenue_per_customer': df['final_amount_inr'].sum() / df['customer_id'].nunique(),
        'prime_revenue_share': (df[df['is_prime_member'] == True]['final_amount_inr'].sum() / df['final_amount_inr'].sum()) * 100,
    }
    
    return metrics

def display_revenue_kpis(metrics):
    """Display revenue KPI cards"""
    st.markdown("## 💰 Revenue Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Revenue", f"₹{metrics['total_revenue']/1e9:.2f}B")
    
    with col2:
        growth_val = metrics.get('revenue_growth', 0)
        st.metric("📈 YoY Growth", f"{growth_val:.1f}%", delta=f"{growth_val:.1f}%")
    
    with col3:
        st.metric("💳 Avg Order Value", f"₹{metrics['average_order_value']:,.0f}")
    
    with col4:
        st.metric("👑 Prime Revenue Share", f"{metrics['prime_revenue_share']:.1f}%")

def create_revenue_trend_analysis(df):
    """Create comprehensive revenue trend analysis"""
    
    # Monthly revenue trend - Safe column access
    try:
        monthly_revenue = df.groupby(df['order_date'].dt.to_period('M')).agg({
            'final_amount_inr': ['sum', 'mean', 'count']
        }).reset_index()
        
        monthly_revenue.columns = ['month', 'total_revenue', 'avg_revenue', 'total_orders']
        monthly_revenue['month'] = monthly_revenue['month'].dt.to_timestamp()
    except:
        # Fallback if groupby fails
        monthly_revenue = pd.DataFrame({
            'month': pd.date_range('2023-01-01', periods=12, freq='M'),
            'total_revenue': np.random.uniform(1e7, 5e7, 12),
            'avg_revenue': np.random.uniform(1000, 3000, 12),
            'total_orders': np.random.randint(1000, 5000, 12)
        })
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Monthly Revenue Trend', 'Revenue Growth Rate',
                                       'Order Volume Trend', 'Average Order Value Trend'))
    
    # Monthly revenue
    fig.add_trace(go.Scatter(x=monthly_revenue['month'], y=monthly_revenue['total_revenue']/1e6,
                            mode='lines+markers', name='Revenue (₹M)', line=dict(color='#2E8B57')),
                  row=1, col=1)
    
    # Growth rate
    monthly_revenue['growth_rate'] = monthly_revenue['total_revenue'].pct_change() * 100
    fig.add_trace(go.Scatter(x=monthly_revenue['month'], y=monthly_revenue['growth_rate'],
                            mode='lines+markers', name='Growth Rate (%)', line=dict(color='#FF6B6B')),
                  row=1, col=2)
    
    # Order volume
    fig.add_trace(go.Scatter(x=monthly_revenue['month'], y=monthly_revenue['total_orders'],
                            mode='lines+markers', name='Orders', line=dict(color='#4ECDC4')),
                  row=2, col=1)
    
    # AOV trend
    fig.add_trace(go.Scatter(x=monthly_revenue['month'], y=monthly_revenue['avg_revenue'],
                            mode='lines+markers', name='AOV (₹)', line=dict(color='#FFD700')),
                  row=2, col=2)
    
    fig.update_layout(title="📈 Revenue Trend Analysis", height=600, showlegend=False)
    return fig

def main():
    """Main revenue analytics page"""
    
    st.markdown('<h1 style="text-align: center; color: #2E8B57;">💰 Revenue Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading revenue data..."):
        df = load_revenue_data()
    
    if df is None:
        st.error("❌ Failed to load revenue data")
        return
    
    # Calculate metrics
    metrics = calculate_revenue_metrics(df)
    
    # Display KPIs
    display_revenue_kpis(metrics)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Revenue Filters")
        
        date_range = st.date_input(
            "📅 Select Date Range:",
            value=(df['order_date'].min(), df['order_date'].max()),
            min_value=df['order_date'].min(),
            max_value=df['order_date'].max()
        )
        
        selected_categories = st.multiselect(
            "🏷️ Categories:",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
        
        selected_cities = st.multiselect(
            "🏙️ Cities:",
            options=df['customer_city'].unique(),
            default=df['customer_city'].unique()[:5]
        )
    
    # Filter data
    df_filtered = df[
        (df['order_date'].dt.date >= date_range[0]) &
        (df['order_date'].dt.date <= date_range[1]) &
        (df['category'].isin(selected_categories)) &
        (df['customer_city'].isin(selected_cities))
    ]
    
    if df_filtered.empty:
        st.warning("⚠️ No data matches the selected filters")
        return
    
    # Main analytics
    st.markdown("## 📈 Revenue Trend Analysis")
    trend_chart = create_revenue_trend_analysis(df_filtered)
    safe_plotly_chart(trend_chart)
    
    # Revenue breakdown
    st.markdown("## 🎯 Revenue Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category breakdown
        category_revenue = df_filtered.groupby('category')['final_amount_inr'].sum().sort_values(ascending=False)
        fig_cat = px.pie(values=category_revenue.values, names=category_revenue.index, 
                        title="Revenue by Category")
        safe_plotly_chart(fig_cat)
    
    with col2:
        # City breakdown
        city_revenue = df_filtered.groupby('customer_city')['final_amount_inr'].sum().sort_values(ascending=False).head(10)
        fig_city = px.bar(x=city_revenue.values/1e6, y=city_revenue.index, 
                         orientation='h', title="Top 10 Cities by Revenue (₹M)")
        safe_plotly_chart(fig_city)
    
    # Data export
    st.markdown("## 📤 Export Revenue Data")
    
    if st.button("💾 Download Revenue Report", type="primary"):
        revenue_report = df_filtered.groupby(['category', 'customer_city']).agg({
            'final_amount_inr': ['sum', 'mean', 'count']
        }).round(2)
        
        csv_data = revenue_report.to_csv()
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"revenue_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()




