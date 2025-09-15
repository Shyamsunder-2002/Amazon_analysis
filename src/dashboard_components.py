"""
Dashboard Components - Reusable dashboard elements
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_metric_cards(metrics_dict, cols=4):
    """Render a row of metric cards"""
    
    columns = st.columns(cols)
    
    for i, (title, data) in enumerate(metrics_dict.items()):
        col_idx = i % cols
        
        with columns[col_idx]:
            if isinstance(data, dict):
                value = data.get('value', 0)
                delta = data.get('delta', None)
                help_text = data.get('help', None)
            else:
                value = data
                delta = None
                help_text = None
            
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )

def render_filter_sidebar(df, filters_config):
    """Render standardized filter sidebar"""
    
    st.sidebar.markdown("## 🎛️ Filters")
    
    applied_filters = {}
    
    for filter_name, filter_config in filters_config.items():
        st.sidebar.markdown(f"### {filter_config['title']}")
        
        if filter_config['type'] == 'multiselect':
            selected = st.sidebar.multiselect(
                filter_config['label'],
                options=filter_config['options'],
                default=filter_config.get('default', filter_config['options'])
            )
            applied_filters[filter_name] = selected
        
        elif filter_config['type'] == 'selectbox':
            selected = st.sidebar.selectbox(
                filter_config['label'],
                options=filter_config['options'],
                index=filter_config.get('default_index', 0)
            )
            applied_filters[filter_name] = selected
        
        elif filter_config['type'] == 'date_range':
            selected = st.sidebar.date_input(
                filter_config['label'],
                value=filter_config.get('default', (df[filter_config['column']].min(), df[filter_config['column']].max()))
            )
            applied_filters[filter_name] = selected
        
        elif filter_config['type'] == 'slider':
            selected = st.sidebar.slider(
                filter_config['label'],
                min_value=filter_config['min_value'],
                max_value=filter_config['max_value'],
                value=filter_config.get('default', filter_config['min_value'])
            )
            applied_filters[filter_name] = selected
    
    return applied_filters

def create_download_section(data_dict, section_title="📤 Export Options"):
    """Create standardized download section"""
    
    st.markdown(f"## {section_title}")
    
    cols = st.columns(len(data_dict))
    
    for i, (button_text, data_info) in enumerate(data_dict.items()):
        with cols[i]:
            if st.button(button_text, type="primary"):
                if isinstance(data_info['data'], pd.DataFrame):
                    csv_data = data_info['data'].to_csv(index=False)
                else:
                    csv_data = data_info['data']
                
                st.download_button(
                    label=f"📥 Download {data_info['filename']}",
                    data=csv_data,
                    file_name=data_info['filename'],
                    mime="text/csv"
                )

def render_status_indicator(status_value, thresholds, labels=None):
    """Render a status indicator with color coding"""
    
    if labels is None:
        labels = ["Poor", "Fair", "Good", "Excellent"]
    
    if status_value >= thresholds[2]:
        status = labels[3]
        color = "🟢"
    elif status_value >= thresholds[1]:
        status = labels[2] 
        color = "🟡"
    elif status_value >= thresholds[0]:
        status = labels[1]
        color = "🟠"
    else:
        status = labels[0]
        color = "🔴"
    
    return f"{color} {status} ({status_value:.1f})"
