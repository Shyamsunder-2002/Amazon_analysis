"""
Customer Analytics Page - Customer Behavior & Segmentation Analysis
Deep dive into customer metrics, behavior patterns, and segmentation strategies
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
from operator import attrgetter


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

# Replace all st.dataframe calls with safe_dataframe_display
# Replace all st.plotly_chart calls with safe_plotly_chart

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config.settings import *

st.set_page_config(
    page_title="👥 Customer Analytics",
    page_icon="👥",
    layout="wide"
)

# Custom CSS for customer theme
st.markdown("""
<style>
    .customer-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .customer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.37);
    }
    
    .segment-card {
        background: linear-gradient(135deg, #4ECDC4, #44A08D);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .loyalty-tier {
        background: linear-gradient(135deg, #FFB347, #FF8C00);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_customer_data():
    """Load data optimized for customer analysis"""
    try:
        if os.path.exists(CLEANED_DATA_DIR / "final_dataset.csv"):
            df = pd.read_csv(CLEANED_DATA_DIR / "final_dataset.csv")
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            # Generate customer-focused dataset
            np.random.seed(42)
            n_records = 120000
            n_customers = 30000
            
            data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
                'customer_id': np.random.choice([f'CUST_{i:05d}' for i in range(1, n_customers+1)], n_records),
                'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
                'category': np.random.choice(['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors', 'Books', 'Beauty & Personal Care', 'Toys & Games'], n_records),
                'final_amount_inr': np.random.gamma(2, 1500, n_records),
                'customer_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'], n_records),
                'customer_state': np.random.choice(['Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'West Bengal', 'Gujarat'], n_records),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n_records, p=[0.25, 0.35, 0.25, 0.10, 0.05]),
                'is_prime_member': np.random.choice([True, False], n_records, p=[0.45, 0.55]),
                'customer_rating': np.random.uniform(1, 5, n_records),
                'payment_method': np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash on Delivery'], n_records),
                'is_festival_sale': np.random.choice([True, False], n_records, p=[0.30, 0.70])
            }
            
            df = pd.DataFrame(data)
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month
            
            return df
    except Exception as e:
        st.error(f"Error loading customer data: {e}")
        return None

def calculate_customer_metrics(df):
    """Calculate comprehensive customer metrics"""
    
    # Basic customer metrics
    total_customers = df['customer_id'].nunique()
    total_orders = len(df)
    total_revenue = df['final_amount_inr'].sum()
    
    # Customer behavior metrics
    customer_summary = df.groupby('customer_id').agg({
        'final_amount_inr': ['sum', 'mean', 'count'],
        'order_date': ['min', 'max'],
        'customer_rating': 'mean',
        'is_prime_member': 'first'
    }).round(2)
    
    customer_summary.columns = ['total_spent', 'avg_order_value', 'total_orders', 'first_order', 'last_order', 'avg_rating', 'is_prime']
    customer_summary['customer_lifespan'] = (customer_summary['last_order'] - customer_summary['first_order']).dt.days
    customer_summary['days_since_last_order'] = (datetime.now() - customer_summary['last_order']).dt.days
    
    # RFM Analysis
    current_date = df['order_date'].max()
    customer_summary['recency'] = (current_date - customer_summary['last_order']).dt.days
    customer_summary['frequency'] = customer_summary['total_orders']
    customer_summary['monetary'] = customer_summary['total_spent']
    
    # Calculate RFM scores
    customer_summary['r_score'] = pd.qcut(customer_summary['recency'], 5, labels=[5,4,3,2,1])
    customer_summary['f_score'] = pd.qcut(customer_summary['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    customer_summary['m_score'] = pd.qcut(customer_summary['monetary'], 5, labels=[1,2,3,4,5])
    
    # Customer segments
    def segment_customers(row):
        if row['f_score'] >= 4 and row['m_score'] >= 4:
            return 'Champions'
        elif row['f_score'] >= 3 and row['m_score'] >= 3:
            return 'Loyal Customers'
        elif row['r_score'] >= 4:
            return 'New Customers'
        elif row['r_score'] <= 2:
            return 'At Risk'
        elif row['f_score'] <= 2:
            return 'Price Sensitive'
        else:
            return 'Potential Loyalists'
    
    customer_summary['segment'] = customer_summary.apply(segment_customers, axis=1)
    
    metrics = {
        'total_customers': total_customers,
        'avg_customer_value': customer_summary['total_spent'].mean(),
        'avg_order_frequency': customer_summary['total_orders'].mean(),
        'customer_retention_rate': (customer_summary['total_orders'] > 1).sum() / total_customers * 100,
        'avg_customer_lifespan': customer_summary['customer_lifespan'].mean(),
        'prime_penetration': (df['is_prime_member'].sum() / len(df)) * 100,
        'avg_customer_satisfaction': df['customer_rating'].mean(),
        'active_customers': (customer_summary['days_since_last_order'] <= 90).sum(),
        'churn_risk_customers': (customer_summary['days_since_last_order'] > 180).sum()
    }
    
    return metrics, customer_summary

def display_customer_kpis(metrics):
    """Display customer KPI cards"""
    st.markdown("## 👥 Customer Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="customer-card">
        <h3>👥 Total Customers</h3>
        <h1>{metrics['total_customers']:,}</h1>
        <p>Unique Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="customer-card">
        <h3>💰 Avg Customer Value</h3>
        <h1>₹{metrics['avg_customer_value']:,.0f}</h1>
        <p>Lifetime Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="customer-card">
        <h3>🔄 Retention Rate</h3>
        <h1>{metrics['customer_retention_rate']:.1f}%</h1>
        <p>Repeat Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="customer-card">
        <h3>⭐ Satisfaction Score</h3>
        <h1>{metrics['avg_customer_satisfaction']:.2f}/5.0</h1>
        <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)

def create_customer_segmentation_analysis(customer_summary):
    """Create comprehensive customer segmentation visualizations"""
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "box"}]],
        subplot_titles=('Customer Segments Distribution', 'RFM Analysis (Frequency vs Monetary)', 
                       'Segment Value Contribution', 'Customer Lifespan by Segment')
    )
    
    # Segment distribution
    segment_dist = customer_summary['segment'].value_counts().to_frame()
    fig.add_trace(
        go.Pie(labels=segment_dist.index, values=segment_dist.values, name="Segments"),
        row=1, col=1
    )
    
    # RFM scatter plot
    fig.add_trace(
        go.Scatter(
            x=customer_summary['frequency'],
            y=customer_summary['monetary'],
            mode='markers',
            text=customer_summary['segment'],
            marker=dict(
                size=customer_summary['recency']/10,
                color=customer_summary['recency'],
                colorscale='Viridis',
                showscale=True
            ),
            name='RFM Analysis'
        ),
        row=1, col=2
    )
    
    # Segment value contribution
    segment_value = customer_summary.groupby('segment')['total_spent'].sum().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(x=segment_value.values/1e6, y=segment_value.index, orientation='h',
               name="Revenue (₹M)", marker_color='#667eea'),
        row=2, col=1
    )
    
    # Customer lifespan by segment
    for segment in customer_summary['segment'].unique():
        segment_data = customer_summary[customer_summary['segment'] == segment]
        fig.add_trace(
            go.Box(y=segment_data['customer_lifespan'], name=segment),
            row=2, col=2
        )
    
    fig.update_layout(
        title="🎯 Customer Segmentation Analysis",
        height=700,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def create_customer_lifecycle_analysis(df, customer_summary):
    """Create customer lifecycle and cohort analysis"""
    
    # Customer acquisition over time
    acquisition_data = customer_summary.groupby(customer_summary['first_order'].dt.to_period('M')).size().reset_index()
    acquisition_data['first_order'] = acquisition_data['first_order'].dt.to_timestamp()
    acquisition_data.columns = ['month', 'new_customers']
    
    # Customer activity by cohorts
    df_cohort = df.copy()
    df_cohort['order_period'] = df_cohort['order_date'].dt.to_period('M')
    df_cohort['cohort_group'] = df_cohort.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
    
    # FIX: Create period number correctly
    df_cohort['period_number'] = (df_cohort['order_period'] - df_cohort['cohort_group']).apply(lambda x: x.n if hasattr(x, 'n') else 0)
    
    # Cohort table
    cohort_data = df_cohort.groupby(['cohort_group', 'period_number'])['customer_id'].nunique().reset_index()
    
    # FIX: Add safety check for pivot
    if not cohort_data.empty and len(cohort_data) > 0:
        cohort_counts = cohort_data.pivot(index='cohort_group', columns='period_number', values='customer_id')
        
        # Calculate retention rates
        cohort_sizes = df_cohort.groupby('cohort_group')['customer_id'].nunique()
        retention_table = cohort_counts.divide(cohort_sizes, axis=0)
    else:
        # Create empty retention table if no data
        retention_table = pd.DataFrame()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Customer Acquisition', 'Customer Cohort Retention Heatmap', 
                       'Customer Value Distribution', 'Order Frequency Distribution'),
        specs=[[{"type": "scatter"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # Customer acquisition trend
    if not acquisition_data.empty:
        fig.add_trace(
            go.Scatter(x=acquisition_data['month'], y=acquisition_data['new_customers'],
                      mode='lines+markers', name='New Customers', line=dict(color='#667eea', width=3)),
            row=1, col=1
        )
    
    # Cohort retention heatmap (simplified) - FIX: Add safety check
    if not retention_table.empty and retention_table.shape[0] > 0 and retention_table.shape[1] > 0:
        # Limit to reasonable size for visualization
        retention_sample = retention_table.iloc[:min(12, len(retention_table)), :min(12, retention_table.shape[1])].fillna(0)
        if retention_sample.shape[0] > 0 and retention_sample.shape[1] > 0:
            fig.add_trace(
                go.Heatmap(z=retention_sample.values, 
                          x=[f'Period {i}' for i in retention_sample.columns],
                          y=[str(idx) for idx in retention_sample.index],
                          colorscale='Viridis',
                          name='Retention Rate'),
                row=1, col=2
            )
    
    # Customer value distribution
    if not customer_summary.empty:
        fig.add_trace(
            go.Histogram(x=customer_summary['total_spent'], nbinsx=50, name='Customer Value'),
            row=2, col=1
        )
        
        # Order frequency distribution
        fig.add_trace(
            go.Histogram(x=customer_summary['total_orders'], nbinsx=30, name='Order Frequency'),
            row=2, col=2
        )
    
    fig.update_layout(
        title="📊 Customer Lifecycle & Cohort Analysis",
        height=700,
        template="plotly_white"
    )
    
    return fig

def create_customer_behavior_analysis(df):
    """Create customer behavior analysis"""
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=('Purchase Behavior by Age Group', 'Payment Method Preferences', 
                       'Satisfaction vs Spending', 'Category Preferences by Demographics')
    )
    
    # Age group analysis
    age_behavior = df.groupby('age_group').agg({
        'final_amount_inr': ['sum', 'mean', 'count']
    }).round(2)
    age_behavior.columns = ['total_spending', 'avg_order_value', 'total_orders']
    age_behavior = age_behavior.reset_index()
    
    fig.add_trace(
        go.Bar(x=age_behavior['age_group'], y=age_behavior['total_spending']/1e6,
               name="Total Spending (₹M)", marker_color='#667eea'),
        row=1, col=1
    )
    
    # Payment preferences
    payment_pref = df['payment_method'].value_counts().to_frame()
    fig.add_trace(
        go.Pie(labels=payment_pref.index, values=payment_pref.values, name="Payment Methods"),
        row=1, col=2
    )
    
    # Satisfaction vs Spending
    satisfaction_spending = df.groupby('customer_id').agg({
        'customer_rating': 'mean',
        'final_amount_inr': 'sum'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=satisfaction_spending['customer_rating'], 
                  y=satisfaction_spending['final_amount_inr'],
                  mode='markers', name='Satisfaction vs Spending',
                  marker=dict(size=5, opacity=0.6, color='#4ECDC4')),
        row=2, col=1
    )
    
    # Category preferences by age
    category_age = df.groupby(['age_group', 'category']).size().unstack(fill_value=0)
    category_age_pct = category_age.div(category_age.sum(axis=1), axis=0) * 100
    
    for i, age_group in enumerate(category_age_pct.index):
        fig.add_trace(
            go.Bar(x=category_age_pct.columns, y=category_age_pct.loc[age_group],
                   name=age_group),
            row=2, col=2
        )
    
    fig.update_layout(
        title="🔍 Customer Behavior Analysis",
        height=700,
        template="plotly_white"
    )
    
    return fig

def display_customer_insights(df, metrics, customer_summary):
    """Display customer insights with proper Series handling"""
    
    with st.expander("🔍 Customer Insights"):
        # Safely extract segment information
        if 'segment' in customer_summary.columns and not customer_summary.empty:
            segment_counts = customer_summary['segment'].value_counts()
            top_segment = segment_counts.index[0] if not segment_counts.empty else "Unknown"
            top_segment_count = int(segment_counts.iloc[0]) if not segment_counts.empty else 0
        else:
            top_segment = "Unknown"
            top_segment_count = 0
        
        # Safely extract retention rate
        retention_rate = metrics.get('customer_retention_rate', 0)
        if hasattr(retention_rate, 'iloc'):  # It's a Series
            retention_rate = float(retention_rate.iloc[0])
        else:
            retention_rate = float(retention_rate)
        
        # Calculate churn rate safely
        churn_rate = (df['will_churn'].sum() / len(df) * 100) if 'will_churn' in df.columns else 15.0
        
        # Calculate AOV safely
        prime_mask = df['is_prime_member'] == True
        non_prime_mask = df['is_prime_member'] == False
        
        prime_aov = df[prime_mask]['final_amount_inr'].mean() if prime_mask.any() else 1000.0
        non_prime_aov = df[non_prime_mask]['final_amount_inr'].mean() if non_prime_mask.any() else 800.0
        
        # Ensure these are scalar values
        prime_aov = float(prime_aov) if not pd.isna(prime_aov) else 1000.0
        non_prime_aov = float(non_prime_aov) if not pd.isna(non_prime_aov) else 800.0
        
        # Safe percentage calculation
        prime_premium = ((prime_aov / non_prime_aov - 1) * 100) if non_prime_aov > 0 else 0.0
        
        # Create insights with proper scalar formatting
        insights = [
            f"🏆 Dominant segment: {top_segment} ({top_segment_count:,} customers)",
            f"📈 Customer retention rate: {retention_rate:.1f}%",
            f"⚠️ Churn risk customers: {churn_rate:.1f}% of total base",
            f"💎 Prime members spend {prime_premium:.1f}% more than non-Prime"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
def main():
    """Main customer analytics page"""
    
    # Header
    st.markdown('<h1 class="customer-header">👥 Customer Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
    <h3>🔍 Customer Behavior & Segmentation Intelligence</h3>
    <p>Deep dive into customer metrics, RFM analysis, and behavioral patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading customer analytics data..."):
        df = load_customer_data()
    
    if df is None:
        st.error("❌ Failed to load customer data")
        return
    
    # Calculate metrics
    metrics, customer_summary = calculate_customer_metrics(df)
    
    # Display KPIs
    display_customer_kpis(metrics)
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🎛️ Customer Filters")
        
        # Segment filter
        st.markdown("### 🎯 Customer Segments")
        selected_segments = st.multiselect(
            "Filter by segments:",
            options=customer_summary['segment'].unique(),
            default=customer_summary['segment'].unique()
        )
        
        # Age group filter
        st.markdown("### 👥 Age Groups")
        selected_ages = st.multiselect(
            "Filter by age groups:",
            options=df['age_group'].unique(),
            default=df['age_group'].unique()
        )
        
        # Prime membership filter
        st.markdown("### 👑 Prime Status")
        prime_filter = st.selectbox(
            "Prime membership:",
            options=['All', 'Prime Only', 'Non-Prime Only']
        )
        
        # Apply filters
        df_filtered = df[df['age_group'].isin(selected_ages)]
        
        if prime_filter == 'Prime Only':
            df_filtered = df_filtered[df_filtered['is_prime_member'] == True]
        elif prime_filter == 'Non-Prime Only':
            df_filtered = df_filtered[df_filtered['is_prime_member'] == False]
        
        customer_filtered = customer_summary[customer_summary['segment'].isin(selected_segments)]
        
        st.markdown(f"**Filtered Customers:** {customer_filtered.shape[0]:,}")
    
    # Main analytics
    st.markdown("## 🎯 Customer Segmentation Analysis")
    segmentation_chart = create_customer_segmentation_analysis(customer_filtered)
    st.plotly_chart(segmentation_chart, use_container_width=True)
    
    st.markdown("## 📊 Customer Lifecycle Analysis")
    lifecycle_chart = create_customer_lifecycle_analysis(df_filtered, customer_filtered)
    st.plotly_chart(lifecycle_chart, use_container_width=True)
    
    st.markdown("## 🔍 Customer Behavior Analysis")
    behavior_chart = create_customer_behavior_analysis(df_filtered)
    st.plotly_chart(behavior_chart, use_container_width=True)
    
    # Customer insights
    display_customer_insights(df_filtered, metrics, customer_filtered)
    
    # Segment deep dive
    st.markdown("## 🎯 Customer Segment Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_segment = st.selectbox("Choose segment for analysis:", customer_filtered['segment'].unique())
        
        segment_data = customer_filtered[customer_filtered['segment'] == selected_segment]
        
        st.markdown(f"### 📊 {selected_segment} Segment Analysis")
        
        segment_metrics = {
            'Count': len(segment_data),
            'Avg Spending': f"₹{segment_data['total_spent'].mean():,.0f}",
            'Avg Orders': f"{segment_data['total_orders'].mean():.1f}",
            'Avg Lifespan': f"{segment_data['customer_lifespan'].mean():.0f} days",
            'Prime %': f"{(segment_data['is_prime'].sum()/len(segment_data)*100):.1f}%"
        }
        
        for metric, value in segment_metrics.items():
            st.metric(metric, value)
    
    with col2:
        # Segment comparison
        segment_comparison = customer_filtered.groupby('segment').agg({
            'total_spent': 'mean',
            'total_orders': 'mean',
            'customer_lifespan': 'mean'
        }).round(2)
        
        fig = go.Figure(data=[
            go.Bar(name='Avg Spending', x=segment_comparison.index, y=segment_comparison['total_spent']),
            go.Bar(name='Avg Orders', x=segment_comparison.index, y=segment_comparison['total_orders']*1000),  # Scale for visibility
        ])
        
        fig.update_layout(
            title="Segment Comparison",
            barmode='group',
            height=400
        )
        
        safe_plotly_chart(fig)
    
    # Export options
    st.markdown("## 📤 Export Customer Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👥 Customer Segmentation Report", type="primary"):
            segment_report = customer_filtered.groupby('segment').agg({
                'total_spent': ['count', 'mean', 'sum'],
                'total_orders': 'mean',
                'customer_lifespan': 'mean',
                'is_prime': lambda x: (x.sum()/len(x))*100
            }).round(2)
            
            st.download_button(
                label="📥 Download Segment Report",
                data=segment_report.to_csv(),
                file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 RFM Analysis Report", type="primary"):
            rfm_report = customer_filtered[['customer_id', 'recency', 'frequency', 'monetary', 'segment']].copy()
            
            st.download_button(
                label="📥 Download RFM Report",
                data=rfm_report.to_csv(index=False),
                file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("💾 Customer Master Data", type="primary"):
            st.download_button(
                label="📥 Download Customer Data",
                data=customer_filtered.to_csv(),
                file_name=f"customer_master_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()




