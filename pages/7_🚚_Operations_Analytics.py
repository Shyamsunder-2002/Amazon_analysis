"""
Operations Analytics Page - Supply Chain & Logistics Analysis
Deep dive into operational metrics, delivery performance, and supply chain insights
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
        if 'use_container_width' in kwargs:
            use_container_width = kwargs.pop('use_container_width')
            kwargs['width'] = "stretch" if use_container_width else "content"
        return st.dataframe(fix_dataframe_for_streamlit(df), **kwargs)
    
    def safe_plotly_chart(fig, **kwargs):
        if 'use_container_width' in kwargs:
            kwargs.pop('use_container_width')
        return st.plotly_chart(fig, **kwargs)

st.set_page_config(
    page_title="🚚 Operations Analytics",
    page_icon="🚚",
    layout="wide"
)

@st.cache_data
def load_operations_data():
    """Load data optimized for operations analysis"""
    try:
        data_path = 'data/raw/amazon_india_complete_2015_2025.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            # Generate operations-focused dataset
            np.random.seed(42)
            n_records = 200000
            
            data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
                'customer_id': [f'CUST_{np.random.randint(1, 50000):05d}' for _ in range(n_records)],
                'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
                'category': np.random.choice(['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors'], n_records),
                'final_amount_inr': np.random.gamma(2, 2000, n_records),
                'delivery_charges': np.random.uniform(0, 300, n_records),
                'delivery_days': np.random.choice(range(0, 15), n_records, p=[0.05, 0.25, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01, 0.005] + [0.001]*6),
                'customer_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata'], n_records),
                'customer_state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'West Bengal'], n_records),
                'payment_method': np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery'], n_records),
                'order_status': np.random.choice(['Delivered', 'In Transit', 'Processing', 'Cancelled', 'Returned'], n_records, p=[0.85, 0.05, 0.03, 0.04, 0.03]),
                'warehouse_location': np.random.choice(['Mumbai_WH', 'Delhi_WH', 'Bangalore_WH', 'Chennai_WH'], n_records),
                'shipping_partner': np.random.choice(['Ekart', 'BlueDart', 'Delhivery', 'FedEx', 'IndiaPost'], n_records),
                'is_prime_member': np.random.choice([True, False], n_records, p=[0.40, 0.60]),
                'customer_rating': np.random.uniform(1, 5, n_records),
                'return_status': np.random.choice(['Not Returned', 'Returned'], n_records, p=[0.90, 0.10])
            }
            
            df = pd.DataFrame(data)
            df['delivery_date'] = df['order_date'] + pd.to_timedelta(df['delivery_days'], unit='D')
            df['on_time_delivery'] = df['delivery_days'] <= 3
            df['order_processing_time'] = np.random.uniform(0.5, 2.0, n_records)
            df['delivery_cost'] = df['delivery_charges'] + np.random.uniform(50, 200, n_records)
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month
            return df
    except Exception as e:
        st.error(f"Error loading operations data: {e}")
        return None

def calculate_operations_metrics(df):
    """Calculate comprehensive operations metrics"""
    
    # Safe metric calculations
    avg_delivery_time = float(safe_column_access(df, 'delivery_days', 3).mean())
    on_time_delivery_rate = float((safe_column_access(df, 'on_time_delivery', True).sum() / len(df)) * 100)
    avg_delivery_cost = float(safe_column_access(df, 'delivery_cost', 100).mean())
    
    # Order fulfillment metrics
    order_status = safe_column_access(df, 'order_status', 'Delivered')
    order_completion_rate = float(((order_status == 'Delivered').sum() / len(df)) * 100)
    cancellation_rate = float(((order_status == 'Cancelled').sum() / len(df)) * 100)
    return_rate = float(((safe_column_access(df, 'return_status', 'Not Returned') == 'Returned').sum() / len(df)) * 100)
    
    # Performance summaries
    try:
        city_performance = df.groupby('customer_city').agg({
            'delivery_days': 'mean',
            'customer_rating': 'mean'
        }).round(2)
    except:
        city_performance = pd.DataFrame()
    
    try:
        warehouse_performance = df.groupby('warehouse_location').agg({
            'delivery_days': 'mean',
            'transaction_id': 'count'
        }).round(2)
        warehouse_performance.columns = ['avg_delivery_days', 'total_orders']
    except:
        warehouse_performance = pd.DataFrame()
    
    try:
        partner_performance = df.groupby('shipping_partner').agg({
            'delivery_days': 'mean',
            'customer_rating': 'mean',
            'transaction_id': 'count'
        }).round(2)
        partner_performance.columns = ['avg_delivery_days', 'avg_rating', 'total_orders']
    except:
        partner_performance = pd.DataFrame()
    
    metrics = {
        'avg_delivery_time': avg_delivery_time,
        'on_time_delivery_rate': on_time_delivery_rate,
        'order_completion_rate': order_completion_rate,
        'cancellation_rate': cancellation_rate,
        'return_rate': return_rate,
        'avg_delivery_cost': avg_delivery_cost,
        'total_orders_processed': len(df),
        'operational_efficiency': (on_time_delivery_rate + order_completion_rate - cancellation_rate - return_rate) / 4
    }
    
    return metrics, city_performance, warehouse_performance, partner_performance

def display_operations_kpis(metrics):
    """Display operations KPI cards"""
    st.markdown("## 🚚 Operations Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Avg Delivery Time", f"{metrics['avg_delivery_time']:.1f} days")
    
    with col2:
        st.metric("✅ On-Time Delivery", f"{metrics['on_time_delivery_rate']:.1f}%")
    
    with col3:
        st.metric("📦 Order Completion", f"{metrics['order_completion_rate']:.1f}%")
    
    with col4:
        st.metric("💰 Avg Delivery Cost", f"₹{metrics['avg_delivery_cost']:.0f}")

def create_delivery_performance_analysis(df, metrics):
    """Create delivery performance analysis"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delivery Time Distribution', 'On-Time Delivery by City',
                       'Delivery Performance Trends', 'Customer Satisfaction vs Delivery Time')
    )
    
    # Delivery time distribution
    delivery_days = safe_column_access(df, 'delivery_days', 3)
    fig.add_trace(
        go.Histogram(x=delivery_days, name='Delivery Days', marker_color='#4ECDC4'),
        row=1, col=1
    )
    
    # On-time delivery by city
    try:
        city_otd = df.groupby('customer_city').apply(
            lambda x: (safe_column_access(x, 'on_time_delivery', True).sum() / len(x)) * 100
        ).sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(x=city_otd.index, y=city_otd.values, name='On-Time %', marker_color='#FFD700'),
            row=1, col=2
        )
    except:
        pass
    
    # Monthly delivery trends
    try:
        monthly_perf = df.groupby(df['order_date'].dt.to_period('M')).apply(
            lambda x: (safe_column_access(x, 'on_time_delivery', True).sum() / len(x)) * 100
        ).reset_index()
        
        if not monthly_perf.empty:
            monthly_perf['order_date'] = monthly_perf['order_date'].dt.to_timestamp()
            fig.add_trace(
                go.Scatter(x=monthly_perf['order_date'], y=monthly_perf.iloc[:, 1],
                          mode='lines+markers', name='Monthly OTD%', line=dict(color='#FF6B6B')),
                row=2, col=1
            )
    except:
        pass
    
    # Satisfaction vs delivery time
    try:
        satisfaction_delivery = df.groupby('delivery_days')['customer_rating'].mean()
        fig.add_trace(
            go.Scatter(x=satisfaction_delivery.index, y=satisfaction_delivery.values,
                      mode='markers', name='Rating vs Delivery', marker=dict(color='#8E44AD', size=8)),
            row=2, col=2
        )
    except:
        pass
    
    fig.update_layout(title="📊 Delivery Performance Analysis", height=700, showlegend=False)
    return fig

def main():
    """Main operations analytics page"""
    
    st.markdown('<h1 style="text-align: center; color: #FF8C00;">🚚 Operations Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading operations data..."):
        df = load_operations_data()
    
    if df is None:
        st.error("❌ Failed to load operations data")
        return
    
    # Calculate metrics
    metrics, city_performance, warehouse_performance, partner_performance = calculate_operations_metrics(df)
    
    # Display KPIs
    display_operations_kpis(metrics)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Operations Filters")
        
        selected_cities = st.multiselect(
            "🏙️ Cities:",
            options=df['customer_city'].unique(),
            default=df['customer_city'].unique()
        )
        
        selected_partners = st.multiselect(
            "🚚 Shipping Partners:",
            options=df['shipping_partner'].unique(),
            default=df['shipping_partner'].unique()
        )
        
        delivery_range = st.slider(
            "📅 Delivery Days Range:",
            min_value=0,
            max_value=15,
            value=(0, 10)
        )
    
    # Filter data
    df_filtered = df[
        (df['customer_city'].isin(selected_cities)) &
        (df['shipping_partner'].isin(selected_partners)) &
        (df['delivery_days'] >= delivery_range[0]) &
        (df['delivery_days'] <= delivery_range[1])
    ]
    
    if df_filtered.empty:
        st.warning("⚠️ No data matches the selected filters")
        return
    
    # Main analytics
    st.markdown("## 📊 Delivery Performance Analysis")
    perf_chart = create_delivery_performance_analysis(df_filtered, metrics)
    safe_plotly_chart(perf_chart)
    
    # Performance tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏙️ City Performance")
        if not city_performance.empty:
            safe_dataframe_display(city_performance.head(10), width="stretch")
        else:
            st.info("No city performance data available")
    
    with col2:
        st.markdown("### 🚚 Shipping Partner Performance")
        if not partner_performance.empty:
            safe_dataframe_display(partner_performance.head(10), width="stretch")
        else:
            st.info("No partner performance data available")

if __name__ == "__main__":
    main()
