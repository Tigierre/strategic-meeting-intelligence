import streamlit as st
import json
from pathlib import Path
import os

st.set_page_config(
    page_title="Strategic Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Strategic Meeting Intelligence")
st.subheader("Transform every conversation into strategic advantage")

@st.cache_data
def load_analysis_data():
    # Prova diversi path possibili
    possible_paths = [
        Path("analysis"),
        Path("./analysis"),
        Path("/app/analysis"),
        Path(os.getcwd()) / "analysis"
    ]
    
    analysis_files = []
    for path in possible_paths:
        if path.exists():
            analysis_files = list(path.glob("analysis_transcription_*.json"))
            if analysis_files:
                break
    
    data = []
    for file in analysis_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception as e:
            st.error(f"Errore caricamento {file}: {e}")
    
    return data

# Main interface
analysis_data = load_analysis_data()

if analysis_data:
    # Metrics overview
    st.header("📊 Intelligence Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_insights = sum(len(d['ai_analysis'].get('insight_strategici', [])) for d in analysis_data)
    total_opportunities = sum(len(d['ai_analysis'].get('opportunita_innovation', [])) for d in analysis_data)
    total_themes = sum(len(d['ai_analysis'].get('temi_ricorrenti', [])) for d in analysis_data)
    
    with col1:
        st.metric("🎙️ Meetings", len(analysis_data))
    with col2:
        st.metric("💡 Insights", total_insights)
    with col3:
        st.metric("🚀 Opportunities", total_opportunities)
    with col4:
        st.metric("🔍 Themes", total_themes)
    
    st.success("✅ System operational with strategic insights generated!")
    st.divider()
    
    # Meeting details
    st.header("📋 Strategic Analysis Results")
    
    for meeting in analysis_data:
        meeting_title = meeting['meeting_info']['title']
        language = meeting['meeting_info']['language'].upper()
        
        with st.expander("🎙️ " + meeting_title + " (" + language + ")"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💡 Strategic Insights")
                insights = meeting['ai_analysis'].get('insight_strategici', [])
                for i, insight in enumerate(insights, 1):
                    st.markdown("**" + str(i) + ". " + insight.get('insight', 'N/A') + "**")
                    if insight.get('azione_suggerita'):
                        st.markdown("   ➡️ Action: " + insight['azione_suggerita'])
                    st.markdown("")
            
            with col2:
                st.subheader("🚀 Innovation Opportunities") 
                opportunities = meeting['ai_analysis'].get('opportunita_innovation', [])
                for i, opp in enumerate(opportunities, 1):
                    impact = opp.get('impatto_potenziale', 'N/A')
                    st.markdown("**" + str(i) + ". " + opp.get('opportunita', 'N/A') + "**")
                    st.markdown("   📈 Impact: " + impact.title())
                    st.markdown("")
            
            # Themes
            st.subheader("🔍 Recurring Themes")
            themes = meeting['ai_analysis'].get('temi_ricorrenti', [])
            for theme in themes:
                importance = theme.get('importanza', 'N/A')
                st.markdown("• **" + theme.get('tema', 'N/A') + "** (Importance: " + importance + ")")

else:
    st.error("⚠️ No analysis data found")
    st.info("📁 Looking for JSON files in analysis/ directory")
    
    # Debug info
    st.write("🔍 Debug Info:")
    current_dir = Path(".")
    st.write(f"Current directory: {current_dir.resolve()}")
    st.write("Files found:")
    for item in current_dir.iterdir():
        st.write(f"  - {item.name}")

st.divider()
st.markdown("🚀 **Powered by Strategic Meeting Intelligence**")
