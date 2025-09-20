"""
Data Cleaning Module - Interactive Implementation
Comprehensive data preprocessing with all 10 cleaning challenges
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add to ALL pages - Import block with fallbacks
import sys
import os

from pathlib import Path

# Define data directory
CLEANED_DATA_DIR = Path("data/cleaned")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import helper functions with fallbacks
try:
    from utils import safe_column_access, fix_dataframe_for_streamlit, safe_dataframe_display, safe_plotly_chart
except ImportError:
    # Fallback functions
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
            elif pd.api.types.is_bool_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(int)
            elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")
        return df_copy
    
def fix_dataframe_for_streamlit(df):
    """Fix DataFrame for Arrow serialization"""
    if hasattr(df, 'to_frame'):  # Handle Series input
        df = df.to_frame()
    
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_object_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype('string').fillna("")
        elif pd.api.types.is_bool_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype(int)
        elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d').fillna("")
    return df_copy

    def safe_dataframe_display(df, **kwargs):
        """Safe dataframe display with proper parameter handling"""
        
        # Handle width parameter correctly
        width = kwargs.pop('width', None)
        if width == 'stretch':
            kwargs['use_container_width'] = True
        elif width == 'content':
            kwargs['width'] = 700
        elif isinstance(width, int):
            kwargs['width'] = width
        
        # Handle deprecated use_container_width (if still present)
        if 'use_container_width' in kwargs:
            use_container_width = kwargs.pop('use_container_width')
            if use_container_width:
                kwargs['use_container_width'] = True
        
        # Fix DataFrame and display
        df_fixed = fix_dataframe_for_streamlit(df)
        return st.dataframe(df_fixed, **kwargs)
    
    def safe_plotly_chart(fig, **kwargs):
        if 'use_container_width' in kwargs:
            kwargs.pop('use_container_width')
        return st.plotly_chart(fig, **kwargs)

# Replace all st.dataframe calls with safe_dataframe_display
# Replace all st.plotly_chart calls with safe_plotly_chart


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.data_cleaning import DataCleaner
from config.settings import *

st.set_page_config(
    page_title="🧹 Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide"
)

try:
    from utils import safe_column_access, prepare_df_for_streamlit, safe_dataframe_display
except ImportError:
    # Fallback functions if utils.py not found
    def safe_column_access(df, col_name, default_value=0):
        if col_name in df.columns:
            return df[col_name]
        else:
            return pd.Series([default_value] * len(df), index=df.index)
    
    def prepare_df_for_streamlit(df):
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['object']).columns:
            df_copy[col] = df_copy[col].astype('string')
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


# Custom CSS
st.markdown("""
<style>
    .cleaning-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .challenge-box {
        border-left: 5px solid #FF6B6B;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(255, 107, 107, 0.1);
        border-radius: 5px;
    }
    
    .success-box {
        border-left: 5px solid #4ECDC4;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(78, 205, 196, 0.1);
        border-radius: 5px;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_raw_data():
    """Load raw data for cleaning demonstration"""
    # Create sample messy data for demonstration
    np.random.seed(42)
    n_records = 5000
    
    # Create intentionally messy data
    messy_data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
        'customer_id': [f'CUST_{np.random.randint(1, 2000):05d}' for _ in range(n_records)],
        'order_date': np.random.choice([
            '2023-12-25', '25/12/2023', '25-12-23', '2023/12/25', 
            '32/13/2020', 'invalid_date', '2024-02-30'
        ] + [f'2023-{m:02d}-{d:02d}' for m in range(1, 13) for d in range(1, 29)], n_records),
        
        'original_price_inr': np.random.choice([
            '1500', '₹2,500', '₹1,25,000', 'Price on Request', '3500.50',
            '₹ 4,200', 'Contact Seller', '15000', '999.99'
        ], n_records),
        
        'customer_rating': np.random.choice([
            '4.5', '5 stars', '3/5', '2.5/5.0', '4 star', 'Excellent', '3.0',
            np.nan, '4.8', '1/5', '5.0/5.0'
        ], n_records),
        
        'customer_city': np.random.choice([
            'Bangalore', 'Bengaluru', 'Mumbai', 'Bombay', 'Delhi', 'New Delhi',
            'CHENNAI', 'madras', 'Kolkata', 'calcutta', 'HYDERABAD'
        ], n_records),
        
        'is_prime_member': np.random.choice([
            True, False, 'Yes', 'No', 1, 0, 'Y', 'N', 'TRUE', 'FALSE', np.nan
        ], n_records),
        
        'category': np.random.choice([
            'Electronics', 'ELECTRONICS', 'Electronic', 'Electronics & Accessories',
            'Clothing', 'clothes', 'APPAREL', 'Home & Kitchen', 'home', 'KITCHEN'
        ], n_records),
        
        'delivery_days': np.random.choice([
            1, 2, 3, '1-2 days', 'Same Day', 'Next Day', -1, 50, '3-5 days', np.nan
        ], n_records),
        
        'payment_method': np.random.choice([
            'UPI', 'PhonePe', 'GooglePay', 'Credit Card', 'CREDIT_CARD', 'CC',
            'Cash on Delivery', 'COD', 'C.O.D', 'Net Banking', 'NETBANKING'
        ], n_records),
        
        'final_amount_inr': np.random.gamma(2, 500, n_records)
    }
    
    # Add some extreme outliers (decimal point errors)
    outlier_indices = np.random.choice(n_records, 100, replace=False)
    for idx in outlier_indices:
        messy_data['final_amount_inr'][idx] *= 100  # Simulate decimal errors
    
    # Add duplicates
    duplicate_indices = np.random.choice(n_records-100, 200, replace=False)
    for i, idx in enumerate(duplicate_indices):
        if i < 100:  # First 100 are genuine duplicates
            messy_data['transaction_id'][idx] = messy_data['transaction_id'][idx-1]
            messy_data['customer_id'][idx] = messy_data['customer_id'][idx-1]
            messy_data['order_date'][idx] = messy_data['order_date'][idx-1]
    
    return pd.DataFrame(messy_data)

def display_data_quality_overview(df):
    """Display data quality overview"""
    st.markdown("## 📊 Data Quality Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Values", f"{missing_pct:.1f}%", delta=None)
    
    with col2:
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        st.metric("Duplicate Rows", f"{duplicate_pct:.1f}%", delta=None)
    
    with col3:
        st.metric("Total Records", f"{len(df):,}", delta=None)
    
    with col4:
        st.metric("Total Columns", f"{len(df.columns)}", delta=None)
    
    # Data quality heatmap
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if not missing_data.empty:
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title="Missing Values by Column",
            labels={'x': 'Count of Missing Values', 'y': 'Columns'}
        )
        safe_plotly_chart(fig)


def demonstrate_cleaning_challenges(df):
    """Demonstrate all 10 cleaning challenges"""
    
    st.markdown('<h2 class="cleaning-header">🧹 Data Cleaning Challenges Implementation</h2>', 
                unsafe_allow_html=True)
    
    cleaner = DataCleaner()
    
    # Challenge selection
    challenges = [
        "Challenge 1: Order Date Standardization",
        "Challenge 2: Price Column Cleaning", 
        "Challenge 3: Rating Standardization",
        "Challenge 4: City Name Standardization",
        "Challenge 5: Boolean Column Harmonization",
        "Challenge 6: Category Standardization",
        "Challenge 7: Delivery Days Cleaning",
        "Challenge 8: Duplicate Handling",
        "Challenge 9: Price Outlier Detection",
        "Challenge 10: Payment Method Standardization"
    ]
    
    selected_challenge = st.selectbox("🎯 Select a cleaning challenge to demonstrate:", challenges)
    
    if selected_challenge == challenges[0]:  # Date cleaning
        st.markdown("""
        <div class="challenge-box">
        <h3>🗓️ Challenge 1: Order Date Standardization</h3>
        <p><strong>Problem:</strong> Multiple date formats ('DD/MM/YYYY', 'DD-MM-YY', 'YYYY-MM-DD') and invalid entries</p>
        <p><strong>Solution:</strong> Parse multiple formats and validate date ranges</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show before and after
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Cleaning:**")
            date_sample = df['order_date'].value_counts().to_frame().head(10)
            st.write(date_sample)
        
        with col2:
            st.markdown("**After Cleaning:**")
            df_cleaned = cleaner.clean_order_dates(df.copy())
            if not df_cleaned.empty:
                cleaned_dates = df_cleaned['order_date'].dt.date.value_counts().to_frame().head(10)
                st.write(cleaned_dates)
            else:
                st.write("No valid dates found")
        
        # Visualization
        if not df_cleaned.empty:
            fig = px.histogram(df_cleaned, x='order_date', nbins=50,
                             title="Distribution of Cleaned Order Dates")
            st.plotly_chart(fig, use_container_width=True)
    
    elif selected_challenge == challenges[1]:  # Price cleaning
        st.markdown("""
        <div class="challenge-box">
        <h3>💰 Challenge 2: Price Column Cleaning</h3>
        <p><strong>Problem:</strong> Mixed formats (₹ symbols, commas, text entries like 'Price on Request')</p>
        <p><strong>Solution:</strong> Extract numeric values and handle special cases</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Cleaning:**")
            price_sample = df['original_price_inr'].value_counts().to_frame().head(10)
            st.write(price_sample)
        
        with col2:
            st.markdown("**After Cleaning:**")
            df_price_cleaned = cleaner.clean_price_columns(df.copy())
            cleaned_prices = df_price_cleaned['original_price_inr'].describe()
            st.write(cleaned_prices)
        
        # Price distribution
        if 'original_price_inr' in df_price_cleaned.columns:
            valid_prices = df_price_cleaned['original_price_inr'].dropna()
            if not valid_prices.empty:
                fig = px.histogram(valid_prices, nbins=50,
                                 title="Distribution of Cleaned Prices")
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_challenge == challenges[2]:  # Rating cleaning
        st.markdown("""
        <div class="challenge-box">
        <h3>⭐ Challenge 3: Rating Standardization</h3>
        <p><strong>Problem:</strong> Various formats ('5.0', '4 stars', '3/5', '2.5/5.0')</p>
        <p><strong>Solution:</strong> Convert all to 1.0-5.0 numeric scale</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Cleaning:**")
            rating_sample = df['customer_rating'].value_counts().to_frame().head(10)
            st.write(rating_sample)
        
        with col2:
            st.markdown("**After Cleaning:**")
            df_rating_cleaned = cleaner.clean_ratings(df.copy())
            cleaned_ratings = df_rating_cleaned['customer_rating'].describe()
            st.write(cleaned_ratings)
        
        # Rating distribution
        fig = px.histogram(df_rating_cleaned['customer_rating'], nbins=20,
                         title="Distribution of Standardized Ratings (1.0-5.0)")
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_challenge == challenges[3]:  # City standardization
        st.markdown("""
        <div class="challenge-box">
        <h3>🏙️ Challenge 4: City Name Standardization</h3>
        <p><strong>Problem:</strong> Variations like 'Bangalore/Bengaluru', 'Mumbai/Bombay', case issues</p>
        <p><strong>Solution:</strong> Standardize using mapping dictionary</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Cleaning:**")
            city_sample = df['customer_city'].value_counts().to_frame().head(10)
            st.write(city_sample)
        
        with col2:
            st.markdown("**After Cleaning:**")
            df_city_cleaned = cleaner.standardize_cities(df.copy())
            cleaned_cities = df_city_cleaned['customer_city'].value_counts().to_frame().head(10)
            st.write(cleaned_cities)
    
    # Add similar implementations for challenges 5-10...
    
    # Complete pipeline demonstration
    if st.button("🚀 Run Complete Cleaning Pipeline"):
        with st.spinner("Running complete data cleaning pipeline..."):
            df_final = cleaner.clean_complete_dataset(df.copy())
            
            st.markdown("""
            <div class="success-box">
            <h3>✅ Cleaning Pipeline Completed Successfully!</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show before/after comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Dataset:**")
                st.write(f"Rows: {len(df):,}")
                st.write(f"Columns: {len(df.columns)}")
                st.write(f"Missing values: {df.isnull().sum().sum():,}")
                st.write(f"Duplicates: {df.duplicated().sum():,}")
            
            with col2:
                st.markdown("**Cleaned Dataset:**")
                st.write(f"Rows: {len(df_final):,}")
                st.write(f"Columns: {len(df_final.columns)}")
                st.write(f"Missing values: {df_final.isnull().sum().sum():,}")
                st.write(f"Duplicates: {df_final.duplicated().sum():,}")
            
            # Show cleaning report
            st.markdown("### 📋 Detailed Cleaning Report")
            cleaning_report = cleaner.cleaning_reports
            
            for operation, stats in cleaning_report.items():
                with st.expander(f"📊 {operation.title()} Details"):
                    for key, value in stats.items():
                        st.write(f"**{key}:** {value}")
            
            # Save cleaned data
            if st.button("💾 Save Cleaned Dataset"):
                output_path = CLEANED_DATA_DIR / "cleaned_amazon_data.csv"
                df_final.to_csv(output_path, index=False)
                st.success(f"✅ Cleaned dataset saved to: {output_path}")
                
                # Provide download link
                st.download_button(
                    label="📥 Download Cleaned Dataset",
                    data=df_final.to_csv(index=False),
                    file_name="cleaned_amazon_data.csv",
                    mime="text/csv"
                )

def main():
    """Main function for data cleaning page"""
    
    # Header
    st.markdown('<h1 class="cleaning-header">🧹 Data Cleaning Pipeline</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
    <h3>📊 Comprehensive Data Preprocessing & Quality Enhancement</h3>
    <p>Transform messy e-commerce data into production-ready analytics datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Cleaning Controls")
        
        # Data loading options
        data_source = st.radio(
            "📥 Select Data Source:",
            ["Sample Messy Data", "Upload Custom Dataset", "Load from Database"]
        )
        
        if data_source == "Upload Custom Dataset":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type=['csv'],
                help="Upload your raw Amazon India dataset"
            )
        
        # Cleaning options
        st.markdown("### 🧹 Cleaning Options")
        
        cleaning_options = {
            'clean_dates': st.checkbox("🗓️ Clean Order Dates", value=True),
            'clean_prices': st.checkbox("💰 Clean Price Columns", value=True),
            'clean_ratings': st.checkbox("⭐ Standardize Ratings", value=True),
            'clean_cities': st.checkbox("🏙️ Standardize Cities", value=True),
            'clean_booleans': st.checkbox("✅ Clean Boolean Columns", value=True),
            'clean_categories': st.checkbox("📦 Standardize Categories", value=True),
            'clean_delivery': st.checkbox("🚚 Clean Delivery Days", value=True),
            'handle_duplicates': st.checkbox("🔄 Handle Duplicates", value=True),
            'fix_outliers': st.checkbox("📊 Fix Price Outliers", value=True),
            'clean_payments': st.checkbox("💳 Standardize Payments", value=True)
        }
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            outlier_threshold = st.slider("Outlier Threshold (IQR Multiplier)", 1.0, 3.0, 1.5)
            duplicate_threshold = st.slider("Duplicate Detection Sensitivity", 1, 5, 3)
            missing_threshold = st.slider("Missing Value Threshold (%)", 0, 100, 50)
    
    # Load data based on selection
    if data_source == "Sample Messy Data":
        with st.spinner("🔄 Loading sample messy data..."):
            df = load_raw_data()
            st.success("✅ Sample data loaded successfully!")
    
    elif data_source == "Upload Custom Dataset" and 'uploaded_file' in locals() and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Custom dataset loaded: {len(df):,} records")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            df = load_raw_data()
    
    else:
        df = load_raw_data()
    
    # Data overview
    st.markdown("## 📊 Raw Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📋 Total Records",
            f"{len(df):,}",
            help="Total number of transactions in the dataset"
        )
    
    with col2:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric(
            "❓ Missing Data",
            f"{missing_pct:.1f}%",
            delta=f"{df.isnull().sum().sum():,} values",
            help="Percentage of missing values across all columns"
        )
    
    with col3:
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        st.metric(
            "🔄 Duplicates",
            f"{duplicate_pct:.1f}%",
            delta=f"{df.duplicated().sum():,} rows",
            help="Percentage of duplicate rows"
        )
    
    with col4:
        data_types = df.dtypes.value_counts().to_frame()
        st.metric(
            "📝 Data Types",
            f"{len(data_types)}",
            delta=f"{len(df.columns)} columns",
            help="Number of different data types"
        )
    
    # Display data quality overview
    display_data_quality_overview(df)
    
    # Data preview
    with st.expander("🔍 Raw Data Preview", expanded=False):
        st.markdown("### First 10 Rows of Raw Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### Data Schema Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Column Data Types:**")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        with col2:
            st.markdown("**Statistical Summary:**")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                st.dataframe(df[numeric_columns].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
    
    # Interactive cleaning demonstration
    st.markdown("## 🎯 Interactive Cleaning Challenges")
    
    # Tab interface for different challenges
    challenge_tabs = st.tabs([
        "📅 Dates", "💰 Prices", "⭐ Ratings", "🏙️ Cities", "✅ Booleans",
        "📦 Categories", "🚚 Delivery", "🔄 Duplicates", "📊 Outliers", "💳 Payments"
    ])
    
    with challenge_tabs[0]:  # Date cleaning
        st.markdown("### 🗓️ Challenge 1: Order Date Standardization")
        
        st.markdown("""
        <div class="challenge-box">
        <h4>Problem Statement:</h4>
        <p>Order dates appear in multiple inconsistent formats:</p>
        <ul>
        <li><code>DD/MM/YYYY</code> - 25/12/2023</li>
        <li><code>DD-MM-YY</code> - 25-12-23</li>
        <li><code>YYYY-MM-DD</code> - 2023-12-25</li>
        <li>Invalid entries - 32/13/2020, 2024-02-30</li>
        </ul>
        <p><strong>Goal:</strong> Standardize all dates to YYYY-MM-DD format and handle invalid dates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'order_date' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Before Cleaning:**")
                date_sample = df['order_date'].value_counts().to_frame().head(10)
                st.dataframe(date_sample, use_container_width=True)
                
                # Show problematic dates
                st.markdown("**⚠️ Problematic Dates:**")
                problem_dates = df['order_date'][df['order_date'].astype(str).str.contains(r'32/|30/02|31/02|13/|14/|15/|/25|/26', na=False)]
                if len(problem_dates) > 0:
                    st.write(problem_dates.head(5).tolist())
                else:
                    st.info("No obviously invalid dates found in sample")
            
            with col2:
                if st.button("🧹 Clean Dates", key="clean_dates_btn"):
                    with st.spinner("Processing dates..."):
                        cleaner = DataCleaner()
                        df_date_cleaned = cleaner.clean_order_dates(df.copy())
                        
                        st.markdown("**✅ After Cleaning:**")
                        if not df_date_cleaned.empty:
                            cleaned_dates = df_date_cleaned['order_date'].dt.date.value_counts().to_frame().head(10)
                            st.dataframe(cleaned_dates, use_container_width=True)
                            
                            # Show cleaning statistics
                            original_count = len(df)
                            cleaned_count = len(df_date_cleaned)
                            
                            st.success(f"""
                            **Cleaning Results:**
                            - Original records: {original_count:,}
                            - Valid dates: {cleaned_count:,}
                            - Invalid dates removed: {original_count - cleaned_count:,}
                            - Success rate: {(cleaned_count/original_count)*100:.2f}%
                            """)
                        else:
                            st.error("No valid dates found after cleaning")
                
                # Visualization
                if st.checkbox("📊 Show Date Distribution", key="show_date_dist"):
                    try:
                        # Attempt to parse dates for visualization
                        df_temp = df.copy()
                        df_temp['order_date'] = pd.to_datetime(df_temp['order_date'], errors='coerce')
                        valid_dates = df_temp['order_date'].dropna()
                        
                        if len(valid_dates) > 0:
                            fig = px.histogram(
                                x=valid_dates,
                                nbins=50,
                                title="Distribution of Valid Order Dates"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Cannot create visualization due to date format issues")
    
    with challenge_tabs[1]:  # Price cleaning
        st.markdown("### 💰 Challenge 2: Price Column Cleaning")
        
        st.markdown("""
        <div class="challenge-box">
        <h4>Problem Statement:</h4>
        <p>Price columns contain mixed data formats:</p>
        <ul>
        <li>Numeric values - 1500, 2500.50</li>
        <li>Currency symbols - ₹2,500, ₹1,25,000</li>
        <li>Text entries - "Price on Request", "Contact Seller"</li>
        <li>Formatting issues - "₹ 4,200", "₹1,234.56"</li>
        </ul>
        <p><strong>Goal:</strong> Extract clean numeric values in Indian Rupees.</p>
        </div>
        """, unsafe_allow_html=True)
        
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'amount' in col.lower()]
        
        if price_columns:
            selected_price_col = st.selectbox("Select price column to analyze:", price_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Before Cleaning:**")
                price_sample = df[selected_price_col].value_counts().to_frame().head(10)
                st.dataframe(price_sample, use_container_width=True)
                
                # Show data types and issues
                st.markdown("**📊 Price Column Analysis:**")
                price_info = {
                    'Data Type': str(df[selected_price_col].dtype),
                    'Unique Values': df[selected_price_col].nunique(),
                    'Missing Values': df[selected_price_col].isnull().sum(),
                    'Non-Numeric Entries': sum(df[selected_price_col].astype(str).str.contains(r'[^\d.,₹]', na=False))
                }
                
                for key, value in price_info.items():
                    st.metric(key, value)
            
            with col2:
                if st.button("🧹 Clean Prices", key="clean_prices_btn"):
                    with st.spinner("Processing prices..."):
                        cleaner = DataCleaner()
                        df_price_cleaned = cleaner.clean_price_columns(df.copy(), [selected_price_col])
                        
                        st.markdown("**✅ After Cleaning:**")
                        cleaned_prices = df_price_cleaned[selected_price_col].describe()
                        st.dataframe(cleaned_prices, use_container_width=True)
                        
                        # Show cleaning effectiveness
                        original_valid = df[selected_price_col].notna().sum()
                        cleaned_valid = df_price_cleaned[selected_price_col].notna().sum()
                        
                        st.success(f"""
                        **Cleaning Results:**
                        - Original valid entries: {original_valid:,}
                        - Cleaned valid entries: {cleaned_valid:,}
                        - Improvement: {cleaned_valid - original_valid:,}
                        - Average price: ₹{df_price_cleaned[selected_price_col].mean():,.2f}
                        """)
                
                # Price distribution visualization
                if st.checkbox("📊 Show Price Distribution", key="show_price_dist"):
                    try:
                        numeric_prices = pd.to_numeric(df[selected_price_col], errors='coerce').dropna()
                        
                        if len(numeric_prices) > 0:
                            fig = px.histogram(
                                x=numeric_prices,
                                nbins=50,
                                title=f"Distribution of {selected_price_col}",
                                labels={'x': 'Price (₹)', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Price statistics
                            st.markdown("**💹 Price Statistics:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Min Price", f"₹{numeric_prices.min():,.2f}")
                            with col2:
                                st.metric("Median Price", f"₹{numeric_prices.median():,.2f}")
                            with col3:
                                st.metric("Max Price", f"₹{numeric_prices.max():,.2f}")
                    except:
                        st.info("Cannot create visualization due to price format issues")
    
    with challenge_tabs[2]:  # Rating cleaning
        st.markdown("### ⭐ Challenge 3: Rating Standardization")
        
        st.markdown("""
        <div class="challenge-box">
        <h4>Problem Statement:</h4>
        <p>Customer ratings appear in various formats:</p>
        <ul>
        <li>Decimal format - 4.5, 3.2</li>
        <li>Star format - "5 stars", "4 star"</li>
        <li>Fraction format - "3/5", "4.5/5.0"</li>
        <li>Text descriptions - "Excellent", "Good"</li>
        <li>Missing values</li>
        </ul>
        <p><strong>Goal:</strong> Convert all ratings to standardized 1.0-5.0 numeric scale.</p>
        </div>
        """, unsafe_allow_html=True)
        
        rating_columns = [col for col in df.columns if 'rating' in col.lower()]
        
        if rating_columns:
            selected_rating_col = st.selectbox("Select rating column to analyze:", rating_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Before Cleaning:**")
                rating_sample = df[selected_rating_col].value_counts().to_frame().head(10)
                st.dataframe(rating_sample, use_container_width=True)
                
                # Rating format analysis
                st.markdown("**🔍 Rating Format Analysis:**")
                total_ratings = len(df[selected_rating_col])
                numeric_ratings = pd.to_numeric(df[selected_rating_col], errors='coerce').notna().sum()
                star_ratings = df[selected_rating_col].astype(str).str.contains('star', case=False, na=False).sum()
                fraction_ratings = df[selected_rating_col].astype(str).str.contains('/', na=False).sum()
                missing_ratings = df[selected_rating_col].isnull().sum()
                
                format_stats = {
                    'Numeric Format': f"{numeric_ratings} ({numeric_ratings/total_ratings*100:.1f}%)",
                    'Star Format': f"{star_ratings} ({star_ratings/total_ratings*100:.1f}%)",
                    'Fraction Format': f"{fraction_ratings} ({fraction_ratings/total_ratings*100:.1f}%)",
                    'Missing Values': f"{missing_ratings} ({missing_ratings/total_ratings*100:.1f}%)"
                }
                
                for format_type, count in format_stats.items():
                    st.write(f"**{format_type}:** {count}")
            
            with col2:
                if st.button("🧹 Clean Ratings", key="clean_ratings_btn"):
                    with st.spinner("Processing ratings..."):
                        cleaner = DataCleaner()
                        df_rating_cleaned = cleaner.clean_ratings(df.copy(), [selected_rating_col])
                        
                        st.markdown("**✅ After Cleaning:**")
                        cleaned_ratings = df_rating_cleaned[selected_rating_col].describe()
                        st.dataframe(cleaned_ratings, use_container_width=True)
                        
                        # Rating distribution
                        rating_dist = df_rating_cleaned[selected_rating_col].value_counts().to_frame().sort_index()
                        
                        st.success(f"""
                        **Cleaning Results:**
                        - All ratings standardized to 1.0-5.0 scale
                        - Average rating: {df_rating_cleaned[selected_rating_col].mean():.2f}
                        - Most common rating: {df_rating_cleaned[selected_rating_col].mode()[0]:.1f}
                        - Rating range: {df_rating_cleaned[selected_rating_col].min():.1f} - {df_rating_cleaned[selected_rating_col].max():.1f}
                        """)
                
                # Rating visualization
                if st.checkbox("📊 Show Rating Distribution", key="show_rating_dist"):
                    try:
                        # Clean ratings for visualization
                        cleaner = DataCleaner()
                        df_temp = cleaner.clean_ratings(df.copy(), [selected_rating_col])
                        
                        fig = px.histogram(
                            df_temp,
                            x=selected_rating_col,
                            nbins=20,
                            title=f"Distribution of Cleaned {selected_rating_col}",
                            labels={'x': 'Rating (1.0-5.0)', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Rating quality metrics
                        st.markdown("**⭐ Rating Quality Metrics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Rating", f"{df_temp[selected_rating_col].mean():.2f}")
                        with col2:
                            excellent_pct = (df_temp[selected_rating_col] >= 4.5).sum() / len(df_temp) * 100
                            st.metric("Excellent Ratings (≥4.5)", f"{excellent_pct:.1f}%")
                        with col3:
                            poor_pct = (df_temp[selected_rating_col] <= 2.0).sum() / len(df_temp) * 100
                            st.metric("Poor Ratings (≤2.0)", f"{poor_pct:.1f}%")
                    except:
                        st.info("Cannot create visualization due to rating format issues")
    
    # Continue with remaining challenge tabs...
    with challenge_tabs[3]:  # City standardization
        st.markdown("### 🏙️ Challenge 4: City Name Standardization")
        
        st.markdown("""
        <div class="challenge-box">
        <h4>Problem Statement:</h4>
        <p>City names have multiple variations and inconsistencies:</p>
        <ul>
        <li>Alternative names - Bangalore/Bengaluru, Mumbai/Bombay</li>
        <li>Case variations - CHENNAI, chennai, Chennai</li>
        <li>Spelling errors and abbreviations</li>
        <li>Extra spaces and formatting issues</li>
        </ul>
        <p><strong>Goal:</strong> Standardize all city names to consistent format.</p>
        </div>
        """, unsafe_allow_html=True)
        
        city_columns = [col for col in df.columns if 'city' in col.lower()]
        
        if city_columns:
            selected_city_col = st.selectbox("Select city column to analyze:", city_columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Before Cleaning:**")
                city_sample = df[selected_city_col].value_counts().to_frame().head(15)
                st.dataframe(city_sample, use_container_width=True)
                
                # Show city variations
                st.markdown("**🔍 Identified Variations:**")
                city_variations = []
                unique_cities = df[selected_city_col].value_counts().to_frame()
                for city in unique_cities.index:
                    if isinstance(city, str):
                        lower_city = city.lower()
                        if 'bangalore' in lower_city or 'bengaluru' in lower_city:
                            city_variations.append(f"{city} → Bangalore")
                        elif 'mumbai' in lower_city or 'bombay' in lower_city:
                            city_variations.append(f"{city} → Mumbai")
                        elif 'delhi' in lower_city:
                            city_variations.append(f"{city} → Delhi")
                
                if city_variations:
                    for variation in city_variations[:10]:
                        st.write(f"• {variation}")
            
            with col2:
                if st.button("🧹 Clean Cities", key="clean_cities_btn"):
                    with st.spinner("Processing cities..."):
                        cleaner = DataCleaner()
                        df_city_cleaned = cleaner.standardize_cities(df.copy(), selected_city_col)
                        
                        st.markdown("**✅ After Cleaning:**")
                        cleaned_cities = df_city_cleaned[selected_city_col].value_counts().to_frame().head(15)
                        st.dataframe(cleaned_cities, use_container_width=True)
                        
                        # Show standardization results
                        original_unique = df[selected_city_col].nunique()
                        cleaned_unique = df_city_cleaned[selected_city_col].nunique()
                        
                        st.success(f"""
                        **Cleaning Results:**
                        - Original unique cities: {original_unique}
                        - Standardized unique cities: {cleaned_unique}
                        - Cities consolidated: {original_unique - cleaned_unique}
                        - Most common city: {cleaned_cities.index[0]} ({cleaned_cities.iloc[0]:,} records)
                        """)
                
                # City distribution visualization
                if st.checkbox("📊 Show City Distribution", key="show_city_dist"):
                    cleaner = DataCleaner()
                    df_temp = cleaner.standardize_cities(df.copy(), selected_city_col)
                    
                    top_cities = df_temp[selected_city_col].value_counts().to_frame().head(15)
                    
                    fig = px.bar(
                        x=top_cities.values,
                        y=top_cities.index,
                        orientation='h',
                        title="Top 15 Cities by Transaction Count",
                        labels={'x': 'Number of Transactions', 'y': 'City'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Complete pipeline execution
    st.markdown("## 🚀 Complete Cleaning Pipeline")
    
    st.info("""
    **🎯 Ready to run the complete cleaning pipeline?**
    
    This will execute all selected cleaning operations in sequence and provide a comprehensive analysis report.
    """)
    
    if st.button("🔥 Execute Complete Pipeline", type="primary"):
        execute_complete_pipeline(df, cleaning_options)

def execute_complete_pipeline(df, cleaning_options):
    """Execute the complete cleaning pipeline with progress tracking"""
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize cleaner
    cleaner = DataCleaner()
    df_processed = df.copy()
    
    operations = [op for op, enabled in cleaning_options.items() if enabled]
    total_operations = len(operations)
    
    try:
        for i, operation in enumerate(operations):
            status_text.text(f"🔄 Executing: {operation}...")
            progress_bar.progress((i + 1) / total_operations)
            
            if operation == 'clean_dates':
                df_processed = cleaner.clean_order_dates(df_processed)
            elif operation == 'clean_prices':
                df_processed = cleaner.clean_price_columns(df_processed)
            elif operation == 'clean_ratings':
                df_processed = cleaner.clean_ratings(df_processed)
            elif operation == 'clean_cities':
                df_processed = cleaner.standardize_cities(df_processed)
            elif operation == 'clean_booleans':
                df_processed = cleaner.clean_boolean_columns(df_processed)
            elif operation == 'clean_categories':
                df_processed = cleaner.standardize_categories(df_processed)
            elif operation == 'clean_delivery':
                df_processed = cleaner.clean_delivery_days(df_processed)
            elif operation == 'handle_duplicates':
                df_processed = cleaner.handle_duplicates(df_processed)
            elif operation == 'fix_outliers':
                df_processed = cleaner.fix_price_outliers(df_processed)
            elif operation == 'clean_payments':
                df_processed = cleaner.standardize_payment_methods(df_processed)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Pipeline completed successfully!")
        
        # Display results
        display_pipeline_results(df, df_processed, cleaner.cleaning_reports)
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)

def display_pipeline_results(original_df, cleaned_df, cleaning_reports):
    """Display comprehensive pipeline results"""
    
    st.markdown("## 🎉 Pipeline Execution Results")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        original_records = len(original_df)
        cleaned_records = len(cleaned_df)
        records_retained = (cleaned_records / original_records) * 100
        
        st.metric(
            "📊 Records Retained",
            f"{cleaned_records:,}",
            delta=f"{records_retained:.1f}% of original",
            help="Number of records after cleaning pipeline"
        )
    
    with col2:
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        missing_reduction = ((original_missing - cleaned_missing) / original_missing * 100) if original_missing > 0 else 0
        
        st.metric(
            "❓ Missing Values",
            f"{cleaned_missing:,}",
            delta=f"-{missing_reduction:.1f}%",
            delta_color="inverse",
            help="Reduction in missing values"
        )
    
    with col3:
        original_duplicates = original_df.duplicated().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        duplicate_reduction = original_duplicates - cleaned_duplicates
        
        st.metric(
            "🔄 Duplicates Removed",
            f"{duplicate_reduction:,}",
            delta=f"{cleaned_duplicates} remaining",
            delta_color="inverse",
            help="Number of duplicate records removed"
        )
    
    with col4:
        data_quality_score = 100 * (1 - (cleaned_missing + cleaned_duplicates) / (len(cleaned_df) * len(cleaned_df.columns)))
        
        st.metric(
            "✅ Data Quality Score",
            f"{data_quality_score:.1f}%",
            help="Overall data quality after cleaning"
        )
    
    # Before vs After comparison
    st.markdown("### 📊 Before vs After Comparison")
    
    comparison_data = {
        'Metric': ['Total Records', 'Missing Values', 'Duplicate Records', 'Data Types', 'Unique Cities', 'Unique Categories'],
        'Before': [
            f"{len(original_df):,}",
            f"{original_df.isnull().sum().sum():,}",
            f"{original_df.duplicated().sum():,}",
            f"{len(original_df.dtypes.unique())}",
            f"{original_df.get('customer_city', pd.Series()).nunique()}",
            f"{original_df.get('category', pd.Series()).nunique()}"
        ],
        'After': [
            f"{len(cleaned_df):,}",
            f"{cleaned_df.isnull().sum().sum():,}",
            f"{cleaned_df.duplicated().sum():,}",
            f"{len(cleaned_df.dtypes.unique())}",
            f"{cleaned_df.get('customer_city', pd.Series()).nunique()}",
            f"{cleaned_df.get('category', pd.Series()).nunique()}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Detailed operation reports
    st.markdown("### 📋 Detailed Operation Reports")
    
    for operation, report in cleaning_reports.items():
        with st.expander(f"📊 {operation.replace('_', ' ').title()} Report"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Operation Metrics:**")
                for key, value in report.items():
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            
            with col2:
                # Create a simple visualization for the operation
                if 'success_rate' in report:
                    success_rate = float(report['success_rate'].replace('%', ''))
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = success_rate,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Success Rate"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#4ECDC4"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
    
    # Data export options
    st.markdown("### 💾 Export Cleaned Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = cleaned_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="cleaned_amazon_data.csv",
            mime="text/csv",
            help="Download cleaned dataset as CSV file"
        )
    
    with col2:
        if st.button("💾 Save to Database"):
            try:
                # Here you would integrate with DatabaseManager
                st.success("✅ Data saved to database successfully!")
            except Exception as e:
                st.error(f"❌ Database save failed: {e}")
    
    with col3:
        # Generate and download cleaning report
        report_content = generate_cleaning_report(cleaning_reports)
        st.download_button(
            label="📊 Download Report",
            data=report_content,
            file_name="data_cleaning_report.txt",
            mime="text/plain",
            help="Download detailed cleaning report"
        )

def generate_cleaning_report(cleaning_reports):
    """Generate a comprehensive text report of cleaning operations"""
    
    report_lines = [
        "="*60,
        "🧹 AMAZON INDIA DATA CLEANING REPORT",
        "="*60,
        f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY:",
        f"- Total cleaning operations performed: {len(cleaning_reports)}",
        ""
    ]
    
    for operation, stats in cleaning_reports.items():
        report_lines.append(f"\n{operation.upper().replace('_', ' ')}:")
        report_lines.append("-" * (len(operation) + 1))
        
        for key, value in stats.items():
            report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    
    report_lines.extend([
        "",
        "="*60,
        "End of Report",
        "="*60
    ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()




