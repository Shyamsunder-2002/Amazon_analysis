"""
Visualization Utilities - Common plotting functions and styling
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color palettes
AMAZON_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
BUSINESS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

def create_kpi_card(title, value, delta=None, prefix="", suffix=""):
    """Create a KPI card with consistent styling"""
    
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta >= 0 else "red"
        delta_symbol = "↗" if delta >= 0 else "↘"
        delta_html = f'<p style="color: {delta_color}; margin: 0;">{delta_symbol} {delta}</p>'
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.37);
    ">
        <h4 style="margin: 0; font-size: 1rem;">{title}</h4>
        <h2 style="margin: 0.5rem 0; font-size: 2rem;">{prefix}{value}{suffix}</h2>
        {delta_html}
    </div>
    """
    
    return card_html

def create_trend_chart(df, x_col, y_col, title="Trend Analysis", color=None):
    """Create a standardized trend chart"""
    
    fig = px.line(df, x=x_col, y=y_col, title=title, color=color)
    
    fig.update_layout(
        template="plotly_white",
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14
    )
    
    return fig

def create_comparison_chart(df, categories, values, chart_type="bar", title="Comparison Analysis"):
    """Create comparison charts (bar, pie, etc.)"""
    
    if chart_type == "bar":
        fig = px.bar(x=categories, y=values, title=title, color=values, color_continuous_scale="Viridis")
    elif chart_type == "pie":
        fig = px.pie(values=values, names=categories, title=title, color_discrete_sequence=AMAZON_COLORS)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=categories, y=values, mode='markers+lines'))
        fig.update_layout(title=title)
    
    fig.update_layout(template="plotly_white")
    return fig

def apply_amazon_theme(fig):
    """Apply Amazon-inspired theme to plotly figures"""
    
    fig.update_layout(
        template="plotly_white",
        font_family="Arial, sans-serif",
        title_font_size=18,
        title_font_color="#333",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=AMAZON_COLORS
    )
    
    return fig




