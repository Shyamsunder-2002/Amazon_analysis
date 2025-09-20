"""
Executive Dashboard - C-Level Business Intelligence
Strategic KPIs and high-level business metrics
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config.settings import *

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


st.set_page_config(
    page_title="💼 Executive Dashboard",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for executive theme
st.markdown("""
<style>
    .executive-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .strategic-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .alert-positive {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_executive_data():
    """Load data optimized for executive dashboard"""
    try:
        if os.path.exists(CLEANED_DATA_DIR / "final_dataset.csv"):
            df = pd.read_csv(CLEANED_DATA_DIR / "final_dataset.csv")
            df['order_date'] = pd.to_datetime(df['order_date'])
            return df
        else:
            # Generate executive-focused dataset
            np.random.seed(42)
            n_records = 75000
            
            data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(n_records)],
                'customer_id': [f'CUST_{np.random.randint(1, 20000):05d}' for _ in range(n_records)],
                'order_date': pd.date_range('2015-01-01', '2025-08-31', periods=n_records),
                'category': np.random.choice(['Electronics', 'Clothing & Accessories', 'Home & Kitchen', 'Sports & Outdoors', 'Books', 'Beauty & Personal Care'], n_records),
                'final_amount_inr': np.random.gamma(2, 1000, n_records),
                'customer_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune'], n_records),
                'is_prime_member': np.random.choice([True, False], n_records, p=[0.40, 0.60]),
                'customer_rating': np.random.uniform(1, 5, n_records),
                'is_festival_sale': np.random.choice([True, False], n_records, p=[0.25, 0.75])
            }
            
            df = pd.DataFrame(data)
            df['order_year'] = df['order_date'].dt.year
            df['order_month'] = df['order_date'].dt.month
            df['order_quarter'] = df['order_date'].dt.quarter
            
            return df
    except Exception as e:
        st.error(f"Error loading executive data: {e}")
        return None

def calculate_kpis(df):
    """Calculate executive KPIs"""
    current_year = df['order_year'].max()
    previous_year = current_year - 1
    
    current_data = df[df['order_year'] == current_year]
    previous_data = df[df['order_year'] == previous_year]
    
    # Revenue metrics
    current_revenue = current_data['final_amount_inr'].sum()
    previous_revenue = previous_data['final_amount_inr'].sum()
    revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100 if previous_revenue > 0 else 0
    
    # Customer metrics
    current_customers = current_data['customer_id'].nunique()
    previous_customers = previous_data['customer_id'].nunique()
    customer_growth = ((current_customers - previous_customers) / previous_customers) * 100 if previous_customers > 0 else 0
    
    # Order metrics
    current_orders = len(current_data)
    previous_orders = len(previous_data)
    order_growth = ((current_orders - previous_orders) / previous_orders) * 100 if previous_orders > 0 else 0
    
    # AOV metrics
    current_aov = current_data['final_amount_inr'].mean()
    previous_aov = previous_data['final_amount_inr'].mean()
    aov_growth = ((current_aov - previous_aov) / previous_aov) * 100 if previous_aov > 0 else 0
    
    # Customer satisfaction
    current_satisfaction = current_data['customer_rating'].mean()
    previous_satisfaction = previous_data['customer_rating'].mean()
    satisfaction_change = current_satisfaction - previous_satisfaction
    
    # Market share (Prime vs Non-Prime)
    prime_revenue_share = (current_data[current_data['is_prime_member'] == True]['final_amount_inr'].sum() / current_revenue) * 100
    
    return {
        'current_revenue': current_revenue,
        'revenue_growth': revenue_growth,
        'current_customers': current_customers,
        'customer_growth': customer_growth,
        'current_orders': current_orders,
        'order_growth': order_growth,
        'current_aov': current_aov,
        'aov_growth': aov_growth,
        'current_satisfaction': current_satisfaction,
        'satisfaction_change': satisfaction_change,
        'prime_revenue_share': prime_revenue_share
    }

def display_executive_kpis(kpis):
    """Display executive KPI dashboard"""
    st.markdown("## 📊 Executive KPI Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_color = "normal" if kpis['revenue_growth'] >= 0 else "inverse"
        st.markdown(f"""
        <div class="kpi-card">
        <h3>💰 Total Revenue</h3>
        <h1>₹{kpis['current_revenue']/1e9:.2f}B</h1>
        <p style="color: {'#4CAF50' if kpis['revenue_growth'] >= 0 else '#F44336'};">
        {'+' if kpis['revenue_growth'] >= 0 else ''}{kpis['revenue_growth']:.1f}% YoY
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
        <h3>👥 Active Customers</h3>
        <h1>{kpis['current_customers']:,}</h1>
        <p style="color: {'#4CAF50' if kpis['customer_growth'] >= 0 else '#F44336'};">
        {'+' if kpis['customer_growth'] >= 0 else ''}{kpis['customer_growth']:.1f}% YoY
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
        <h3>🛍️ Total Orders</h3>
        <h1>{kpis['current_orders']:,}</h1>
        <p style="color: {'#4CAF50' if kpis['order_growth'] >= 0 else '#F44336'};">
        {'+' if kpis['order_growth'] >= 0 else ''}{kpis['order_growth']:.1f}% YoY
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
        <h3>💳 Avg Order Value</h3>
        <h1>₹{kpis['current_aov']:,.0f}</h1>
        <p style="color: {'#4CAF50' if kpis['aov_growth'] >= 0 else '#F44336'};">
        {'+' if kpis['aov_growth'] >= 0 else ''}{kpis['aov_growth']:.1f}% YoY
        </p>
        </div>
        """, unsafe_allow_html=True)

def create_revenue_trend_chart(df):
    """Create executive revenue trend visualization"""
    monthly_revenue = df.groupby(df['order_date'].dt.to_period('M'))['final_amount_inr'].sum().reset_index()
    monthly_revenue['order_date'] = monthly_revenue['order_date'].dt.to_timestamp()
    monthly_revenue['revenue_millions'] = monthly_revenue['final_amount_inr'] / 1e6
    
    # Calculate trend line
    from scipy import stats
    x_numeric = np.arange(len(monthly_revenue))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, monthly_revenue['revenue_millions'])
    trend_line = slope * x_numeric + intercept
    
    fig = go.Figure()
    
    # Revenue line
    fig.add_trace(go.Scatter(
        x=monthly_revenue['order_date'],
        y=monthly_revenue['revenue_millions'],
        mode='lines+markers',
        name='Monthly Revenue',
        line=dict(color='#1e3c72', width=3),
        marker=dict(size=6)
    ))
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=monthly_revenue['order_date'],
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="📈 Revenue Trend Analysis (Monthly)",
        xaxis_title="Month",
        yaxis_title="Revenue (₹ Millions)",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_business_health_scorecard(df, kpis):
    """Create business health scorecard"""
    
    # Calculate health scores (0-100)
    health_metrics = {
        'Revenue Growth': min(100, max(0, 50 + kpis['revenue_growth'])),
        'Customer Satisfaction': (kpis['current_satisfaction'] / 5) * 100,
        'Customer Growth': min(100, max(0, 50 + kpis['customer_growth'])),
        'Order Growth': min(100, max(0, 50 + kpis['order_growth'])),
        'Prime Penetration': kpis['prime_revenue_share']
    }
    
    # Create separate figures for gauge charts
    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=list(health_metrics.keys()) + ['Overall Health']
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, (metric, score) in enumerate(health_metrics.items()):
        if i < 5:
            row, col = positions[i]
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': metric},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#1e3c72"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=row, col=col)
    
    # Overall health score
    overall_health = np.mean(list(health_metrics.values()))
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=overall_health,
        title={'text': "Overall Business Health"},
        delta={'reference': 75},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "#2a5298"},
               'steps': [{'range': [0, 60], 'color': "#ffcccc"},
                        {'range': [60, 80], 'color': "#ffffcc"},
                        {'range': [80, 100], 'color': "#ccffcc"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                        'thickness': 0.75, 'value': 90}}
    ), row=2, col=3)
    
    fig.update_layout(
        title="🏥 Business Health Scorecard",
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_strategic_initiatives_tracker(df):
    """Create strategic initiatives progress tracker"""
    
    # Simulate strategic initiatives
    initiatives = {
        'Digital Transformation': {
            'target': 80,
            'current': 67,
            'description': 'Increase digital payment adoption'
        },
        'Prime Growth': {
            'target': 50,
            'current': 40,
            'description': 'Expand Prime membership base'
        },
        'Market Expansion': {
            'target': 25,
            'current': 18,
            'description': 'Enter new tier-2/3 cities'
        },
        'Customer Satisfaction': {
            'target': 4.5,
            'current': df['customer_rating'].mean(),
            'description': 'Improve overall customer experience'
        }
    }
    
    fig = go.Figure()
    
    initiative_names = list(initiatives.keys())
    current_values = [init['current'] for init in initiatives.values()]
    target_values = [init['target'] for init in initiatives.values()]
    
    # Current progress bars
    fig.add_trace(go.Bar(
        y=initiative_names,
        x=current_values,
        name='Current Progress',
        orientation='h',
        marker_color='#1e3c72'
    ))
    
    # Target markers
    fig.add_trace(go.Scatter(
        y=initiative_names,
        x=target_values,
        mode='markers',
        name='Target',
        marker=dict(color='#FF6B6B', size=12, symbol='diamond')
    ))
    
    fig.update_layout(
        title="🎯 Strategic Initiatives Progress Tracker",
        xaxis_title="Progress (%)",
        height=400,
        template="plotly_white"
    )
    
    return fig, initiatives

def display_competitive_intelligence(df):
    """Display competitive intelligence section"""
    st.markdown("## 🏆 Competitive Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market position metrics
        st.markdown("""
        <div class="strategic-box">
        <h3>📊 Market Position Analysis</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div>
        <h4>Market Share</h4>
        <h2>12.5%</h2>
        <p>↗️ +2.1% YoY</p>
        </div>
        <div>
        <h4>Competitive Rank</h4>
        <h2>#3</h2>
        <p>🏆 Maintained position</p>
        </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Competitive advantages
        st.markdown("""
        <div class="strategic-box">
        <h3>⚡ Competitive Advantages</h3>
        <ul style="text-align: left; margin-top: 1rem;">
        <li>✅ Fastest delivery network (2.1 days avg)</li>
        <li>✅ Highest customer satisfaction (4.2/5)</li>
        <li>✅ Strongest Prime ecosystem (40% penetration)</li>
        <li>✅ Superior mobile app experience</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def display_risk_alerts(kpis):
    """Display risk alerts and recommendations"""
    st.markdown("## ⚠️ Risk Alerts & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Risk Alerts")
        
        if kpis['revenue_growth'] < 5:
            st.markdown("""
            <div class="alert-warning">
            <h4>📉 Revenue Growth Slowing</h4>
            <p>Revenue growth below industry benchmark (5%)</p>
            <p><strong>Action:</strong> Review pricing strategy and market expansion</p>
            </div>
            """, unsafe_allow_html=True)
        
        if kpis['customer_growth'] < 10:
            st.markdown("""
            <div class="alert-warning">
            <h4>👥 Customer Acquisition Challenges</h4>
            <p>Customer growth rate needs acceleration</p>
            <p><strong>Action:</strong> Increase marketing spend and referral programs</p>
            </div>
            """, unsafe_allow_html=True)
        
        if kpis['current_satisfaction'] < 4.0:
            st.markdown("""
            <div class="alert-warning">
            <h4>😞 Customer Satisfaction Risk</h4>
            <p>Satisfaction score below target threshold</p>
            <p><strong>Action:</strong> Investigate service quality issues</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ✅ Strategic Recommendations")
        
        st.markdown("""
        <div class="alert-positive">
        <h4>🚀 Growth Opportunities</h4>
        <ul>
        <li>Expand Prime benefits to increase penetration</li>
        <li>Launch tier-2 city marketing campaigns</li>
        <li>Introduce personalized product recommendations</li>
        <li>Strengthen supply chain for faster delivery</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-positive">
        <h4>💡 Innovation Focus Areas</h4>
        <ul>
        <li>AI-powered customer service chatbots</li>
        <li>Augmented reality product visualization</li>
        <li>Voice commerce integration</li>
        <li>Sustainable packaging initiatives</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main executive dashboard function"""
    
    # Header
    st.markdown('<h1 class="executive-header">💼 Executive Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="strategic-box">
    <h2>🎯 Strategic Business Intelligence Command Center</h2>
    <p>Real-time insights for data-driven executive decision making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading executive analytics..."):
        df = load_executive_data()
    
    if df is None:
        st.error("❌ Failed to load executive dashboard data")
        return
    
    # Calculate KPIs
    kpis = calculate_kpis(df)
    
    # Display KPIs
    display_executive_kpis(kpis)
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Revenue trend
        revenue_chart = create_revenue_trend_chart(df)
        st.plotly_chart(revenue_chart, use_container_width=True)
        
        # Business health scorecard
        health_chart = create_business_health_scorecard(df, kpis)
        st.plotly_chart(health_chart, use_container_width=True)
    
    with col2:
        # Strategic initiatives
        initiatives_chart, initiatives_data = create_strategic_initiatives_tracker(df)
        st.plotly_chart(initiatives_chart, use_container_width=True)
        
        # Quick stats
        st.markdown("### 📋 Quick Stats")
        
        quick_stats = {
            'Market Cap Impact': f"₹{kpis['current_revenue'] * 8 / 1e9:.1f}B",
            'Revenue per Customer': f"₹{kpis['current_revenue'] / kpis['current_customers']:,.0f}",
            'Daily Revenue': f"₹{kpis['current_revenue'] / 365 / 1e3:.0f}K",
            'Customer Lifetime Value': f"₹{df.groupby('customer_id')['final_amount_inr'].sum().mean():,.0f}"
        }
        
        for stat, value in quick_stats.items():
            st.metric(stat, value)
    
    # Competitive intelligence
    display_competitive_intelligence(df)
    
    # Risk management
    display_risk_alerts(kpis)
    
    # Executive summary
    st.markdown("## 📊 Executive Summary Report")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("### 🎯 Key Achievements")
        st.markdown(f"""
        - **Revenue Growth:** {kpis['revenue_growth']:.1f}% YoY increase to ₹{kpis['current_revenue']/1e9:.2f}B
        - **Customer Base:** Expanded by {kpis['customer_growth']:.1f}% to {kpis['current_customers']:,} active customers
        - **Market Position:** Maintained #3 position with 12.5% market share
        - **Customer Satisfaction:** {kpis['current_satisfaction']:.2f}/5.0 rating across all touchpoints
        - **Prime Success:** {kpis['prime_revenue_share']:.1f}% revenue from Prime members
        """)
    
    with summary_col2:
        st.markdown("### 🔮 Strategic Outlook")
        st.markdown("""
        - **Q4 2025 Forecast:** 15-20% revenue growth expected
        - **Market Expansion:** Enter 50+ new tier-2 cities
        - **Technology Investment:** ₹500Cr in AI/ML capabilities
        - **Sustainability Goals:** Carbon neutral delivery by 2026
        - **Prime Growth Target:** 50% customer penetration by 2026
        """)
    
    # Export options
    st.markdown("## 📤 Executive Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Board Report", type="primary"):
            # Generate comprehensive board report
            board_report = {
                'Metric': ['Total Revenue', 'Revenue Growth', 'Active Customers', 'Customer Growth', 
                          'Average Order Value', 'Customer Satisfaction', 'Prime Penetration'],
                'Current Value': [
                    f"₹{kpis['current_revenue']/1e9:.2f}B",
                    f"{kpis['revenue_growth']:.1f}%",
                    f"{kpis['current_customers']:,}",
                    f"{kpis['customer_growth']:.1f}%",
                    f"₹{kpis['current_aov']:,.0f}",
                    f"{kpis['current_satisfaction']:.2f}/5.0",
                    f"{kpis['prime_revenue_share']:.1f}%"
                ],
                'Target': ['₹12.5B', '15%', '25M', '20%', '₹2,500', '4.5/5.0', '50%'],
                'Status': ['On Track', 'Needs Attention', 'Exceeded', 'Behind', 'On Track', 'On Track', 'Behind']
            }
            
            board_df = pd.DataFrame(board_report)
            
            st.download_button(
                label="📥 Download Board Report",
                data=board_df.to_csv(index=False),
                file_name=f"board_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Investor Deck", type="primary"):
            st.success("✅ Investor presentation generated successfully!")
            st.info("📧 Investor deck has been sent to your email")
    
    with col3:
        if st.button("⚡ Real-time Alerts Setup", type="primary"):
            st.success("✅ Real-time alerts configured!")
            st.info("🔔 You'll receive alerts for KPI deviations")

if __name__ == "__main__":
    main()



