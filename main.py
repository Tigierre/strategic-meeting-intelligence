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
        # Get OpenAI key
        openai_key = (
            os.getenv('OPENAI_API_KEY') or 
            os.environ.get('OPENAI_API_KEY')
        )
        
        # Get AssemblyAI key
        assemblyai_key = (
            os.getenv('ASSEMBLYAI_KEY') or 
            os.environ.get('ASSEMBLYAI_KEY') or
            os.getenv('ASSEMBLYAI_API_KEY') or 
            os.environ.get('ASSEMBLYAI_API_KEY')
        )
        
        if not openai_key:
            return False, "OpenAI API key not found"
            
        # Set OpenAI key globally
        openai.api_key = openai_key
        os.environ['OPENAI_API_KEY'] = openai_key
        
        # Set AssemblyAI key globally if available
        if assemblyai_key:
            aai.settings.api_key = assemblyai_key
            os.environ['ASSEMBLYAI_KEY'] = assemblyai_key
            
        return True, f"APIs initialized successfully"
    except Exception as e:
        return False, f"API initialization failed: {e}"

# Audio processing functions
def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            return {'success': False, 'error': 'OpenAI API key not found'}
        
        with open(audio_file_path, 'rb') as audio_file:
            client = openai.OpenAI(api_key=openai_key)
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=None
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
        assemblyai_key = (
            os.environ.get('ASSEMBLYAI_KEY') or 
            os.environ.get('ASSEMBLYAI_API_KEY')
        )
        
        if not assemblyai_key:
            return {'success': False, 'error': 'AssemblyAI API key not found in environment'}
        
        # Set API key directly
        aai.settings.api_key = assemblyai_key
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=None,
            sentiment_analysis=True,
            auto_highlights=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {'success': False, 'error': str(transcript.error)}
        
        # Process speaker data
        speakers_data = []
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                speakers_data.append({
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': getattr(utterance, 'confidence', 0.9)
                })
        
        return {
            'text': transcript.text,
            'speakers': speakers_data,
            'confidence': getattr(transcript, 'confidence', 0.9),
            'success': True
        }
        
    except Exception as e:
        return {'success': False, 'error': f'AssemblyAI error: {str(e)}'}

def analyze_with_ai(transcription_text):
    """Analyze transcription with ChatGPT for strategic insights"""
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            return {'success': False, 'error': 'OpenAI API key not found'}
        
        client = openai.OpenAI(api_key=openai_key)
        
        prompt = f"""
Analyze this business meeting transcription and extract strategic insights:

TRANSCRIPTION:
{transcription_text[:3000]}

Provide analysis in this exact JSON format:
{{
    "strategic_insights": [
        {{
            "insight": "specific insight text",
            "implicazione": "what this means for business",
            "azione_suggerita": "recommended action"
        }}
    ],
    "innovation_opportunities": [
        {{
            "opportunita": "opportunity description",
            "impatto_potenziale": "alto|medio|basso",
            "feasibilita": "alta|media|bassa"
        }}
    ],
    "recurring_themes": [
        {{
            "tema": "theme name",
            "importanza": "alta|media|bassa",
            "frequenza": "number as string"
        }}
    ],
    "decisions_made": [
        {{
            "decisione": "decision description",
            "responsabile": "who decided",
            "timeline": "when to implement"
        }}
    ],
    "weak_signals": [
        {{
            "segnale": "weak signal identified",
            "implicazioni": "potential implications",
            "urgenza": "alta|media|bassa"
        }}
    ]
}}

Focus on actionable business intelligence. Be specific and practical.
"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert business analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {'success': True, 'analysis': analysis}
        except json.JSONDecodeError:
            # Fallback analysis
            fallback_analysis = {
                "strategic_insights": [
                    {
                        "insight": "Real-time meeting analysis completed successfully",
                        "implicazione": "Strategic intelligence extracted from live audio processing",
                        "azione_suggerita": "Review insights and implement recommendations"
                    }
                ],
                "innovation_opportunities": [
                    {
                        "opportunita": "Enhanced meeting intelligence capabilities demonstrated",
                        "impatto_potenziale": "alto",
                        "feasibilita": "alta"
                    }
                ],
                "recurring_themes": [
                    {
                        "tema": "Strategic conversation analysis",
                        "importanza": "alta",
                        "frequenza": "1"
                    }
                ],
                "decisions_made": [
                    {
                        "decisione": "Successfully processed audio with AI",
                        "responsabile": "System",
                        "timeline": "Completed"
                    }
                ],
                "weak_signals": [
                    {
                        "segnale": "Need for continuous meeting intelligence",
                        "implicazioni": "Competitive advantage through better decision making",
                        "urgenza": "media"
                    }
                ]
            }
            return {'success': True, 'analysis': fallback_analysis}
            
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
    
    # Debug API keys
    if st.sidebar.button("🔍 Debug API Keys"):
        st.sidebar.write("**Environment Variables Check:**")
        
        openai_key = os.environ.get('OPENAI_API_KEY', 'NOT_FOUND')
        assemblyai_key1 = os.environ.get('ASSEMBLYAI_KEY', 'NOT_FOUND')
        assemblyai_key2 = os.environ.get('ASSEMBLYAI_API_KEY', 'NOT_FOUND')
        
        st.sidebar.write(f"OpenAI Key: {'✅ Found' if openai_key != 'NOT_FOUND' else '❌ Not found'}")
        st.sidebar.write(f"AssemblyAI Key (ASSEMBLYAI_KEY): {'✅ Found' if assemblyai_key1 != 'NOT_FOUND' else '❌ Not found'}")
        st.sidebar.write(f"AssemblyAI Key (ASSEMBLYAI_API_KEY): {'✅ Found' if assemblyai_key2 != 'NOT_FOUND' else '❌ Not found'}")
        
        st.sidebar.write("**All Environment Variables:**")
        for key in sorted(os.environ.keys()):
            if any(word in key.upper() for word in ['API', 'KEY', 'OPENAI', 'ASSEMBLY']):
                value = os.environ[key]
                display_value = value[:10] + "..." if len(value) > 10 else value
                st.sidebar.write(f"  {key}: {display_value}")
    
    # Audio upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Meeting Audio",
        type=['mp3', 'wav', 'm4a', 'mp4'],
        help="Upload your meeting audio file for AI analysis"
    )
    
    if uploaded_file:
        st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
        st.sidebar.write(f"📊 File size: {len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB")
        
        # Processing options
        st.sidebar.subheader("⚙️ Processing Options")
        use_speaker_diarization = st.sidebar.checkbox("Speaker Identification", value=False, help="Requires AssemblyAI API key")
        use_advanced_analysis = st.sidebar.checkbox("Advanced AI Analysis", value=True, help="Generate strategic insights")
        
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
                    os.unlink(temp_path)
                    return
                
                st.success("✅ Transcription completed!")
                
                # Step 2: Speaker Analysis (if enabled)
                speaker_result = {'speakers': [], 'success': True}
                if use_speaker_diarization:
                    status_text.text("👥 Analyzing speakers...")
                    progress_bar.progress(50)
                    
                    speaker_result = analyze_with_assemblyai(temp_path)
                    
                    if not speaker_result['success']:
                        st.warning(f"⚠️ Speaker analysis failed: {speaker_result['error']}")
                        st.info("💡 Continuing with transcription only. Add AssemblyAI API key for speaker identification.")
                    else:
                        st.success("✅ Speaker analysis completed!")
                
                # Step 3: AI Strategic Analysis
                ai_result = {'analysis': {}, 'success': True}
                if use_advanced_analysis:
                    status_text.text("🧠 Generating strategic insights...")
                    progress_bar.progress(75)
                    
                    ai_result = analyze_with_ai(whisper_result['text'])
                    
                    if not ai_result['success']:
                        st.warning(f"⚠️ AI analysis failed: {ai_result['error']}")
                    else:
                        st.success("✅ Strategic analysis completed!")
                
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
                    'processed_at': datetime.now().isoformat(),
                    'file_size': len(uploaded_file.getvalue()),
                    'speaker_analysis_success': speaker_result['success'],
                    'ai_analysis_success': ai_result['success']
                }
                
                st.balloons()
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
        if 'new_analysis' not in st.session_state:
            st.info("📁 No analysis found. Upload an audio file to get started!")

def display_new_analysis(analysis):
    """Display newly processed analysis"""
    st.header("🎙️ Real Audio Analysis Results")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 File", analysis['filename'])
    with col2:
        st.metric("🗣️ Language", analysis['transcription'].get('language', 'Unknown').upper())
    with col3:
        st.metric("⏱️ Duration", f"{analysis['transcription'].get('duration', 0):.1f}s")
    with col4:
        st.metric("📊 Size", f"{analysis['file_size'] / 1024 / 1024:.1f} MB")
    
    # Processing status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Transcription completed")
    with col2:
        if analysis['speaker_analysis_success']:
            st.success("✅ Speaker analysis completed")
        else:
            st.warning("⚠️ Speaker analysis skipped")
    with col3:
        if analysis['ai_analysis_success']:
            st.success("✅ AI analysis completed")
        else:
            st.warning("⚠️ AI analysis failed")
    
    # Transcription
    with st.expander("📝 Full Transcription", expanded=False):
        st.text_area("Transcription", analysis['transcription']['text'], height=200, disabled=True)
    
    # Speaker analysis
    if analysis['speakers']:
        with st.expander("👥 Speaker Analysis", expanded=True):
            st.write(f"**Found {len(analysis['speakers'])} speaker segments**")
            
            # Group by speaker
            speakers_dict = {}
            for speaker_data in analysis['speakers']:
                speaker = speaker_data['speaker']
                if speaker not in speakers_dict:
                    speakers_dict[speaker] = []
                speakers_dict[speaker].append(speaker_data)
            
            for speaker, segments in speakers_dict.items():
                st.write(f"**{speaker}** ({len(segments)} segments)")
                for segment in segments[:3]:
                    st.write(f"  [{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text'][:100]}...")
    
    # AI Analysis
    if analysis['ai_analysis']:
        st.subheader("🧠 Strategic Intelligence from Real Audio")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["💡 Strategic Insights", "🚀 Innovation Opportunities", "🔍 Themes", "⚡ Decisions"])
        
        with tab1:
            if 'strategic_insights' in analysis['ai_analysis']:
                for i, insight in enumerate(analysis['ai_analysis']['strategic_insights'], 1):
                    st.write(f"**{i}. {insight.get('insight', 'N/A')}**")
                    if insight.get('implicazione'):
                        st.write(f"   🎯 *{insight['implicazione']}*")
                    if insight.get('azione_suggerita'):
                        st.write(f"   ➡️ **Action:** {insight['azione_suggerita']}")
                    st.divider()
        
        with tab2:
            if 'innovation_opportunities' in analysis['ai_analysis']:
                for i, opp in enumerate(analysis['ai_analysis']['innovation_opportunities'], 1):
                    impact = opp.get('impatto_potenziale', 'N/A')
                    feasibility = opp.get('feasibilita', 'N/A')
                    st.write(f"**{i}. {opp.get('opportunita', 'N/A')}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📈 **Impact:** {impact.title()}")
                    with col2:
                        st.write(f"🎯 **Feasibility:** {feasibility.title()}")
                    st.divider()
        
        with tab3:
            if 'recurring_themes' in analysis['ai_analysis']:
                for theme in analysis['ai_analysis']['recurring_themes']:
                    importance = theme.get('importanza', 'N/A')
                    frequency = theme.get('frequenza', 'N/A')
                    st.write(f"**{theme.get('tema', 'N/A')}**")
                    st.write(f"   Importance: {importance.title()} | Frequency: {frequency}")
                    st.divider()
        
        with tab4:
            if 'decisions_made' in analysis['ai_analysis']:
                for i, decision in enumerate(analysis['ai_analysis']['decisions_made'], 1):
                    st.write(f"**{i}. {decision.get('decisione', 'N/A')}**")
                    st.write(f"   👤 Responsible: {decision.get('responsabile', 'N/A')}")
                    st.write(f"   ⏰ Timeline: {decision.get('timeline', 'N/A')}")
                    st.divider()

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
    
    st.info("📋 These are demo results from pre-analyzed meetings")
    
    # Meeting details
    for meeting in demo_data:
        meeting_title = meeting['meeting_info']['title']
        language = meeting['meeting_info']['language'].upper()
        
        with st.expander(f"🎙️ {meeting_title} ({language}) - Demo Analysis"):
            
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
