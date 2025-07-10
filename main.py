import streamlit as st
import json
import os
import tempfile
import time
from pathlib import Path
import requests
import openai
import assemblyai as aai
from datetime import datetime
import uuid

# Page config
st.set_page_config(
    page_title="Strategic Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)

# Initialize APIs
@st.cache_resource
def init_apis():
    try:
        openai_key = os.getenv('OPENAI_API_KEY', '')
        assemblyai_key = os.getenv('ASSEMBLYAI_API_KEY', '')
        
        if openai_key:
            openai.api_key = openai_key
            
        if assemblyai_key:
            aai.settings.api_key = assemblyai_key
            
        return True, "APIs initialized"
    except Exception as e:
        return False, f"API initialization failed: {e}"

# Audio processing functions
def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            client = openai.OpenAI()
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=None  # Auto-detect
            )
            
            return {
                'text': response.text,
                'language': getattr(response, 'language', 'unknown'),
                'duration': getattr(response, 'duration', 0),
                'success': True
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_with_assemblyai(audio_file_path):
    """Advanced analysis with AssemblyAI including speaker diarization"""
    try:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=None,  # Auto-detect number of speakers
            sentiment_analysis=True,
            auto_highlights=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {'success': False, 'error': transcript.error}
        
        # Process speaker data
        speakers_data = []
        if transcript.utterances:
            for utterance in transcript.utterances:
                speakers_data.append({
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': utterance.confidence
                })
        
        return {
            'text': transcript.text,
            'speakers': speakers_data,
            'confidence': transcript.confidence,
            'success': True
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_with_ai(transcription_text):
    """Analyze transcription with ChatGPT for strategic insights"""
    try:
        client = openai.OpenAI()
        
        prompt = f"""
Analyze this business meeting transcription and extract strategic insights:

TRANSCRIPTION:
{transcription_text}

Provide analysis in JSON format with:
- recurring_themes (tema, importanza, frequenza)
- strategic_insights (insight, implicazione, azione_suggerita)
- innovation_opportunities (opportunita, impatto_potenziale, feasibilita)
- decisions_made (decisione, responsabile, timeline)
- weak_signals (segnale, implicazioni, urgenza)

Focus on actionable business intelligence.
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert business analyst specializing in strategic meeting intelligence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {'success': True, 'analysis': analysis}
        except json.JSONDecodeError:
            return {'success': False, 'error': 'Invalid JSON response from AI'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Load demo data function
@st.cache_data
def load_demo_data():
    """Load existing demo analysis data"""
    analysis_path = Path("analysis")
    if not analysis_path.exists():
        return []
    
    data = []
    analysis_files = list(analysis_path.glob("analysis_transcription_*.json"))
    
    for file in analysis_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception:
            continue
    
    return data

# Main app
def main():
    st.title("🧠 Strategic Meeting Intelligence")
    st.subheader("Transform every conversation into strategic advantage")
    
    # Initialize APIs
    api_status, api_message = init_apis()
    
    # Sidebar
    st.sidebar.header("🎙️ Audio Processing")
    
    # Audio upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Meeting Audio",
        type=['mp3', 'wav', 'm4a', 'mp4'],
        help="Upload your meeting audio file for AI analysis"
    )
    
    if uploaded_file:
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Processing options
        st.sidebar.subheader("⚙️ Processing Options")
        use_speaker_diarization = st.sidebar.checkbox("Speaker Identification", value=True)
        use_advanced_analysis = st.sidebar.checkbox("Advanced AI Analysis", value=True)
        
        # Process button
        if st.sidebar.button("🚀 Process Audio", type="primary"):
            if not api_status:
                st.error(f"❌ API Error: {api_message}")
                st.stop()
            
            # Processing workflow
            with st.spinner("🎙️ Processing audio file..."):
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Whisper Transcription
                status_text.text("🔄 Transcribing with Whisper AI...")
                progress_bar.progress(25)
                
                whisper_result = transcribe_with_whisper(temp_path)
                
                if not whisper_result['success']:
                    st.error(f"❌ Transcription failed: {whisper_result['error']}")
                    return
                
                # Step 2: Speaker Analysis (if enabled)
                if use_speaker_diarization:
                    status_text.text("👥 Analyzing speakers...")
                    progress_bar.progress(50)
                    
                    speaker_result = analyze_with_assemblyai(temp_path)
                    
                    if not speaker_result['success']:
                        st.warning(f"⚠️ Speaker analysis failed: {speaker_result['error']}")
                        speaker_result = {'speakers': [], 'success': False}
                else:
                    speaker_result = {'speakers': [], 'success': True}
                
                # Step 3: AI Strategic Analysis
                if use_advanced_analysis:
                    status_text.text("🧠 Generating strategic insights...")
                    progress_bar.progress(75)
                    
                    ai_result = analyze_with_ai(whisper_result['text'])
                    
                    if not ai_result['success']:
                        st.error(f"❌ AI analysis failed: {ai_result['error']}")
                        return
                else:
                    ai_result = {'analysis': {}, 'success': True}
                
                # Step 4: Complete
                status_text.text("✅ Processing complete!")
                progress_bar.progress(100)
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Store results in session state
                st.session_state.new_analysis = {
                    'filename': uploaded_file.name,
                    'transcription': whisper_result,
                    'speakers': speaker_result.get('speakers', []),
                    'ai_analysis': ai_result.get('analysis', {}),
                    'processed_at': datetime.now().isoformat()
                }
                
                st.success("🎉 Audio processing completed successfully!")
                st.rerun()
    
    # Display results
    if 'new_analysis' in st.session_state:
        display_new_analysis(st.session_state.new_analysis)
    
    # Load and display demo data
    demo_data = load_demo_data()
    
    if demo_data:
        st.header("📊 Previous Analysis Results")
        display_demo_analysis(demo_data)
    else:
        st.info("📁 No previous analysis found. Upload an audio file to get started!")

def display_new_analysis(analysis):
    """Display newly processed analysis"""
    st.header("🎙️ New Analysis Results")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 File", analysis['filename'])
    with col2:
        st.metric("🗣️ Language", analysis['transcription'].get('language', 'Unknown').upper())
    with col3:
        st.metric("⏱️ Duration", f"{analysis['transcription'].get('duration', 0):.1f}s")
    
    # Transcription
    with st.expander("📝 Full Transcription", expanded=False):
        st.text_area("Transcription", analysis['transcription']['text'], height=200)
    
    # Speaker analysis
    if analysis['speakers']:
        with st.expander("👥 Speaker Analysis", expanded=True):
            for i, speaker_data in enumerate(analysis['speakers'][:10]):  # Show first 10
                st.write(f"**{speaker_data['speaker']}** ({speaker_data['start']:.1f}s - {speaker_data['end']:.1f}s)")
                st.write(f"  {speaker_data['text']}")
    
    # AI Analysis
    if analysis['ai_analysis']:
        st.subheader("🧠 Strategic Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'strategic_insights' in analysis['ai_analysis']:
                st.write("**💡 Strategic Insights:**")
                for insight in analysis['ai_analysis']['strategic_insights']:
                    st.write(f"• {insight.get('insight', 'N/A')}")
                    if insight.get('azione_suggerita'):
                        st.write(f"  ➡️ Action: {insight['azione_suggerita']}")
        
        with col2:
            if 'innovation_opportunities' in analysis['ai_analysis']:
                st.write("**🚀 Innovation Opportunities:**")
                for opp in analysis['ai_analysis']['innovation_opportunities']:
                    st.write(f"• {opp.get('opportunita', 'N/A')}")
                    st.write(f"  📈 Impact: {opp.get('impatto_potenziale', 'N/A')}")

def display_demo_analysis(demo_data):
    """Display demo analysis data"""
    
    # Metrics overview
    total_insights = sum(len(d['ai_analysis'].get('insight_strategici', [])) for d in demo_data)
    total_opportunities = sum(len(d['ai_analysis'].get('opportunita_innovation', [])) for d in demo_data)
    total_themes = sum(len(d['ai_analysis'].get('temi_ricorrenti', [])) for d in demo_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎙️ Demo Meetings", len(demo_data))
    with col2:
        st.metric("💡 Demo Insights", total_insights)
    with col3:
        st.metric("🚀 Demo Opportunities", total_opportunities)
    with col4:
        st.metric("🔍 Demo Themes", total_themes)
    
    # Meeting details
    for meeting in demo_data:
        meeting_title = meeting['meeting_info']['title']
        language = meeting['meeting_info']['language'].upper()
        
        with st.expander(f"🎙️ {meeting_title} ({language}) - Demo"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💡 Strategic Insights")
                insights = meeting['ai_analysis'].get('insight_strategici', [])
                for i, insight in enumerate(insights, 1):
                    st.write(f"**{i}. {insight.get('insight', 'N/A')}**")
                    if insight.get('azione_suggerita'):
                        st.write(f"   ➡️ Action: {insight['azione_suggerita']}")
            
            with col2:
                st.subheader("🚀 Innovation Opportunities") 
                opportunities = meeting['ai_analysis'].get('opportunita_innovation', [])
                for i, opp in enumerate(opportunities, 1):
                    impact = opp.get('impatto_potenziale', 'N/A')
                    st.write(f"**{i}. {opp.get('opportunita', 'N/A')}**")
                    st.write(f"   📈 Impact: {impact.title()}")

if __name__ == "__main__":
    main()
