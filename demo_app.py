
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Strategic Meeting Intelligence Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Strategic Meeting Intelligence")
st.subheader("Transform every conversation into strategic advantage")

# Carica dati
@st.cache_data
def load_analysis_data():
    analysis_dir = Path("/content/strategic_meeting_dataset/analysis")
    analysis_files = list(analysis_dir.glob("analysis_*.json"))
    
    data = []
    for file in analysis_files:
        with open(file, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

# Main interface
try:
    analysis_data = load_analysis_data()
    
    # Metrics overview
    st.header("📊 Intelligence Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_insights = sum(len(d['ai_analysis'].get('insight_strategici', [])) for d in analysis_data)
    total_opportunities = sum(len(d['ai_analysis'].get('opportunita_innovation', [])) for d in analysis_data)
    total_themes = sum(len(d['ai_analysis'].get('temi_ricorrenti', [])) for d in analysis_data)
    
    with col1:
        st.metric("Meetings Analyzed", len(analysis_data))
    with col2:
        st.metric("Strategic Insights", total_insights)
    with col3:
        st.metric("Innovation Opportunities", total_opportunities)
    with col4:
        st.metric("Recurring Themes", total_themes)
    
    # Meeting details
    st.header("📋 Meeting Analysis Details")
    
    for i, meeting in enumerate(analysis_data):
        with st.expander(f"🎙️ {meeting['meeting_info']['title']}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💡 Strategic Insights")
                insights = meeting['ai_analysis'].get('insight_strategici', [])
                for insight in insights:
                    st.write(f"• **{insight.get('insight', 'N/A')}**")
                    st.write(f"  ↳ Action: {insight.get('azione_suggerita', 'N/A')}")
            
            with col2:
                st.subheader("🚀 Innovation Opportunities") 
                opportunities = meeting['ai_analysis'].get('opportunita_innovation', [])
                for opp in opportunities:
                    impact = opp.get('impatto_potenziale', 'N/A')
                    st.write(f"• **{opp.get('opportunita', 'N/A')}** (Impact: {impact})")
            
            # Themes
            st.subheader("🔍 Recurring Themes")
            themes = meeting['ai_analysis'].get('temi_ricorrenti', [])
            for theme in themes:
                importance = theme.get('importanza', 'N/A')
                st.write(f"• {theme.get('tema', 'N/A')} (Importance: {importance})")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.write("Make sure the analysis files are available in /content/strategic_meeting_dataset/analysis/")

st.markdown("---")
st.markdown("🚀 **Powered by Strategic Meeting Intelligence** - Transform conversations into competitive advantage")
