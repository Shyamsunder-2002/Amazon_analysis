"""
Amazon India Analytics Platform - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from streamlit_utils import prepare_df_for_streamlit, safe_column_access, display_dataframe_safe
except ImportError:
    # Fallback functions if utils not found
    def prepare_df_for_streamlit(df):
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['object']).columns:
            df_copy[col] = df_copy[col].astype('string')
        return df_copy
    
    def safe_column_access(df, column_name, default_value=0):
        if column_name in df.columns:
            return df[column_name]
        else:
            return pd.Series([default_value] * len(df), index=df.index)

# Page configuration
st.set_page_config(
    page_title="🛒 Amazon India Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data"""
    try:
        data_path = Path('data/raw/amazon_india_complete_2015_2025.csv')
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            st.error("❌ Data file not found. Please run setup_project.py first.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🛒 Amazon India: A Decade of Sales Analytics</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
    <h3>📊 Comprehensive E-Commerce Analytics Platform (2015-2025)</h3>
    <p>Advanced data processing, interactive dashboards, and AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        df = load_sample_data()
    
    if df is not None:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📊 Total Revenue</h3>
            <h2>₹{df['final_amount_inr'].sum()/1e9:.2f}B</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>🛍️ Total Orders</h3>
            <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
            <h3>👥 Unique Customers</h3>
            <h2>{df['customer_id'].nunique():,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
            <h3>💰 Avg Order Value</h3>
            <h2>₹{df['final_amount_inr'].mean():,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("## 🚀 Getting Started")
        
        st.info("""
        **📋 Explore the Analytics Platform:**
        
        1. **🧹 Data Cleaning**: Advanced data preprocessing pipeline
        2. **📊 EDA Analysis**: 20+ comprehensive analytical visualizations  
        3. **💼 Executive Dashboard**: Strategic KPIs and business metrics
        4. **💰 Revenue Analytics**: Financial performance and forecasting
        5. **👥 Customer Analytics**: Segmentation and behavior analysis
        6. **📦 Product Analytics**: Catalog performance insights
        7. **🚚 Operations Analytics**: Supply chain and logistics
        8. **🔮 Advanced Analytics**: ML models and predictions
        
        **👈 Use the sidebar to navigate between different modules!**
        """)
        
        # Quick data preview
        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head(), width="stretch")
    
    else:
        st.error("⚠️ Please run the setup script first to generate sample data")
        st.code("python setup_project.py", language="bash")

if __name__ == "__main__":
    main()




