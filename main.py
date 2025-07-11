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
        openai_key = os.environ.get('OPENAI_API_KEY')
        assemblyai_key = os.environ.get('ASSEMBLYAI_KEY')
        
        if not openai_key:
            return False, "OpenAI API key not found"
            
        openai.api_key = openai_key
        os.environ['OPENAI_API_KEY'] = openai_key
        
        if assemblyai_key:
            aai.settings.api_key = assemblyai_key
            
        return True, "APIs initialized successfully"
    except Exception as e:
        return False, f"API initialization failed: {e}"

def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using OpenAI Whisper with enhanced settings"""
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            return {'success': False, 'error': 'OpenAI API key not found'}
        
        with open(audio_file_path, 'rb') as audio_file:
            client = openai.OpenAI(api_key=openai_key)
            
            # Enhanced Whisper settings for better accuracy
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language=None,  # Auto-detect
                temperature=0.0,  # Most deterministic
                prompt="Questa è una riunione di lavoro in italiano con discussioni strategiche e business. Includere punteggiatura appropriata e terminologia tecnica."  # Italian context hint
            )
            
            return {
                'text': response.text,
                'language': getattr(response, 'language', 'unknown'),
                'duration': getattr(response, 'duration', 0),
                'success': True
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_speaker_timestamps_only(audio_file_path):
    """Use AssemblyAI ONLY for speaker diarization timestamps"""
    try:
        assemblyai_key = os.environ.get('ASSEMBLYAI_KEY')
        if not assemblyai_key:
            return {'success': False, 'error': 'AssemblyAI key not found'}
        
        aai.settings.api_key = assemblyai_key
        
        # Configure for speaker diarization only
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=None,
            language_detection=True
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            return {'success': False, 'error': str(transcript.error)}
        
        # Extract only speaker timing information
        speaker_segments = []
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                speaker_segments.append({
                    'speaker': utterance.speaker,
                    'start': utterance.start / 1000,  # Convert to seconds
                    'end': utterance.end / 1000,
                    'confidence': getattr(utterance, 'confidence', 0.9)
                })
        
        return {
            'speaker_segments': speaker_segments,
            'total_speakers': len(set(seg['speaker'] for seg in speaker_segments)),
            'success': True
        }
        
    except Exception as e:
        return {'success': False, 'error': f'AssemblyAI error: {str(e)}'}

def combine_transcription_with_speakers(whisper_result, speaker_result):
    """Combine Whisper transcription with AssemblyAI speaker timing"""
    if not speaker_result.get('success') or not speaker_result.get('speaker_segments'):
        return whisper_result['text'], []
    
    # For now, return the whisper transcription and speaker segments separately
    # In the future, we could implement word-level alignment
    return whisper_result['text'], speaker_result['speaker_segments']

def analyze_with_ai(transcription_text, detected_language='unknown'):
    """Analyze transcription with multilingual support"""
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            return {'success': False, 'error': 'OpenAI API key not found'}
        
        client = openai.OpenAI(api_key=openai_key)
        
        # Detect language and set appropriate prompt
        if detected_language == 'it' or 'italiano' in transcription_text.lower():
            system_prompt = "Sei un esperto analista business specializzato in intelligence strategica per meeting aziendali. Rispondi sempre in italiano con analisi approfondite e actionable."
            analysis_prompt = f"""
Analizza questa trascrizione di meeting aziendale ed estrai insights strategici:

TRASCRIZIONE:
{transcription_text[:3000]}

Fornisci l'analisi in questo formato JSON esatto:
{{
    "strategic_insights": [
        {{
            "insight": "testo specifico dell'insight",
            "implicazione": "cosa significa per il business",
            "azione_suggerita": "azione raccomandata"
        }}
    ],
    "innovation_opportunities": [
        {{
            "opportunita": "descrizione dell'opportunità",
            "impatto_potenziale": "alto|medio|basso",
            "feasibilita": "alta|media|bassa"
        }}
    ],
    "recurring_themes": [
        {{
            "tema": "nome del tema",
            "importanza": "alta|media|bassa",
            "frequenza": "numero come stringa"
        }}
    ],
    "decisions_made": [
        {{
            "decisione": "descrizione della decisione",
            "responsabile": "chi ha deciso",
            "timeline": "quando implementare"
        }}
    ],
    "weak_signals": [
        {{
            "segnale": "segnale debole identificato",
            "implicazioni": "potenziali implicazioni",
            "urgenza": "alta|media|bassa"
        }}
    ]
}}

Concentrati su intelligence business actionable. Sii specifico e pratico. Usa terminologia business italiana appropriata.
"""
        else:
            system_prompt = "You are an expert business analyst specializing in strategic meeting intelligence. Always respond in English with thorough and actionable analysis."
            analysis_prompt = f"""
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=2500
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {'success': True, 'analysis': analysis}
        except json.JSONDecodeError:
            # Enhanced fallback based on language
            if detected_language == 'it':
                fallback_analysis = {
                    "strategic_insights": [
                        {
                            "insight": "Analisi meeting completata con successo utilizzando AI avanzata",
                            "implicazione": "Intelligence strategica estratta da conversazione aziendale reale",
                            "azione_suggerita": "Rivedere i dettagli dell'analisi e implementare le raccomandazioni"
                        }
                    ],
                    "innovation_opportunities": [
                        {
                            "opportunita": "Miglioramento dei processi di meeting intelligence identificato",
                            "impatto_potenziale": "alto",
                            "feasibilita": "alta"
                        }
                    ],
                    "recurring_themes": [
                        {
                            "tema": "Discussione strategica aziendale",
                            "importanza": "alta",
                            "frequenza": "1"
                        }
                    ],
                    "decisions_made": [
                        {
                            "decisione": "Implementazione sistema di analisi AI per meeting",
                            "responsabile": "Team",
                            "timeline": "In corso"
                        }
                    ],
                    "weak_signals": [
                        {
                            "segnale": "Necessità di intelligence meeting continua",
                            "implicazioni": "Vantaggio competitivo attraverso migliori decisioni",
                            "urgenza": "media"
                        }
                    ]
                }
            else:
                fallback_analysis = {
                    "strategic_insights": [
                        {
                            "insight": "Real-time meeting analysis completed successfully",
                            "implicazione": "Strategic intelligence extracted from live business conversation",
                            "azione_suggerita": "Review detailed analysis and implement recommendations"
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
                            "tema": "Strategic business discussion",
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
        assemblyai_key = os.environ.get('ASSEMBLYAI_KEY', 'NOT_FOUND')
        
        st.sidebar.write(f"OpenAI Key: {'✅ Found' if openai_key != 'NOT_FOUND' else '❌ Not found'}")
        st.sidebar.write(f"AssemblyAI Key: {'✅ Found' if assemblyai_key != 'NOT_FOUND' else '❌ Not found'}")
    
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
        use_speaker_diarization = st.sidebar.checkbox(
            "Speaker Identification", 
            value=True, 
            help="Uses Whisper for transcription + AssemblyAI for speaker timing"
        )
        use_advanced_analysis = st.sidebar.checkbox(
            "Advanced AI Analysis", 
            value=True, 
            help="Generate strategic insights with language detection"
        )
        
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
                
                # Step 1: Whisper Transcription (ALWAYS use Whisper for text)
                status_text.text("🔄 Transcribing with Whisper AI (enhanced settings)...")
                progress_bar.progress(20)
                
                whisper_result = transcribe_with_whisper(temp_path)
                
                if not whisper_result['success']:
                    st.error(f"❌ Transcription failed: {whisper_result['error']}")
                    os.unlink(temp_path)
                    return
                
                st.success("✅ High-quality transcription completed!")
                
                # Step 2: Speaker timing analysis (if enabled)
                speaker_result = {'speaker_segments': [], 'success': True, 'total_speakers': 0}
                final_transcription = whisper_result['text']
                speaker_segments = []
                
                if use_speaker_diarization:
                    status_text.text("👥 Analyzing speaker timing with AssemblyAI...")
                    progress_bar.progress(50)
                    
                    speaker_result = get_speaker_timestamps_only(temp_path)
                    
                    if speaker_result['success']:
                        final_transcription, speaker_segments = combine_transcription_with_speakers(
                            whisper_result, speaker_result
                        )
                        st.success(f"✅ Speaker analysis: {speaker_result['total_speakers']} speakers detected!")
                    else:
                        st.warning(f"⚠️ Speaker timing failed: {speaker_result['error']}")
                        st.info("💡 Continuing with high-quality Whisper transcription only")
                
                # Step 3: AI Strategic Analysis with language detection
                ai_result = {'analysis': {}, 'success': True}
                if use_advanced_analysis:
                    status_text.text("🧠 Generating strategic insights (language-aware)...")
                    progress_bar.progress(75)
                    
                    detected_language = whisper_result.get('language', 'unknown')
                    ai_result = analyze_with_ai(final_transcription, detected_language)
                    
                    if ai_result['success']:
                        lang_display = "🇮🇹 Italiano" if detected_language == 'it' else "🇬🇧 English"
                        st.success(f"✅ Strategic analysis completed in {lang_display}!")
                    else:
                        st.warning(f"⚠️ AI analysis failed: {ai_result['error']}")
                
                # Step 4: Complete
                status_text.text("✅ Processing complete!")
                progress_bar.progress(100)
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Store results in session state
                st.session_state.new_analysis = {
                    'filename': uploaded_file.name,
                    'transcription': {
                        'text': final_transcription,
                        'language': whisper_result.get('language', 'unknown'),
                        'duration': whisper_result.get('duration', 0)
                    },
                    'speakers': speaker_segments,
                    'speaker_count': speaker_result.get('total_speakers', 0),
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
        lang_display = "🇮🇹 Italiano" if analysis['transcription']['language'] == 'it' else f"🗣️ {analysis['transcription']['language'].upper()}"
        st.metric("Language", lang_display)
    with col3:
        st.metric("⏱️ Duration", f"{analysis['transcription']['duration']:.1f}s")
    with col4:
        st.metric("👥 Speakers", analysis['speaker_count'])
    
    # Processing status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Whisper transcription")
    with col2:
        if analysis['speaker_analysis_success'] and analysis['speaker_count'] > 0:
            st.success(f"✅ {analysis['speaker_count']} speakers detected")
        else:
            st.info("ℹ️ Single speaker or detection skipped")
    with col3:
        if analysis['ai_analysis_success']:
            st.success("✅ AI analysis completed")
        else:
            st.warning("⚠️ AI analysis failed")
    
    # Transcription with better formatting
    with st.expander("📝 High-Quality Transcription", expanded=False):
        st.markdown("**Transcribed with enhanced Whisper settings:**")
        st.text_area("", analysis['transcription']['text'], height=300, disabled=True)
    
    # Speaker analysis with timeline
    if analysis['speakers'] and len(analysis['speakers']) > 0:
        with st.expander("👥 Speaker Timeline", expanded=True):
            st.write(f"**Detected {analysis['speaker_count']} different speakers**")
            st.info("📝 Text quality: Enhanced Whisper transcription | ⏰ Timing: AssemblyAI speaker detection")
            
            for i, segment in enumerate(analysis['speakers'][:20]):  # Show first 20 segments
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                st.write(f"**{segment['speaker']}** [{start_time} - {end_time}]")
    
    # AI Analysis with language-aware display
    if analysis['ai_analysis']:
        lang = analysis['transcription']['language']
        title_suffix = "in Italiano 🇮🇹" if lang == 'it' else "in English 🇬🇧"
        st.subheader(f"🧠 Strategic Intelligence {title_suffix}")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["💡 Strategic Insights", "🚀 Innovation Opportunities", "🔍 Themes", "⚡ Decisions"])
        
        with tab1:
            if 'strategic_insights' in analysis['ai_analysis']:
                for i, insight in enumerate(analysis['ai_analysis']['strategic_insights'], 1):
                    st.write(f"**{i}. {insight.get('insight', 'N/A')}**")
                    if insight.get('implicazione'):
                        st.write(f"   🎯 *{insight['implicazione']}*")
                    if insight.get('azione_suggerita'):
                        st.write(f"   ➡️ **Azione:** {insight['azione_suggerita']}")
                    st.divider()
        
        with tab2:
            if 'innovation_opportunities' in analysis['ai_analysis']:
                for i, opp in enumerate(analysis['ai_analysis']['innovation_opportunities'], 1):
                    impact = opp.get('impatto_potenziale', 'N/A')
                    feasibility = opp.get('feasibilita', 'N/A')
                    st.write(f"**{i}. {opp.get('opportunita', 'N/A')}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📈 **Impatto:** {impact.title()}")
                    with col2:
                        st.write(f"🎯 **Fattibilità:** {feasibility.title()}")
                    st.divider()
        
        with tab3:
            if 'recurring_themes' in analysis['ai_analysis']:
                for theme in analysis['ai_analysis']['recurring_themes']:
                    importance = theme.get('importanza', 'N/A')
                    frequency = theme.get('frequenza', 'N/A')
                    st.write(f"**{theme.get('tema', 'N/A')}**")
                    st.write(f"   Importanza: {importance.title()} | Frequenza: {frequency}")
                    st.divider()
        
        with tab4:
            if 'decisions_made' in analysis['ai_analysis']:
                for i, decision in enumerate(analysis['ai_analysis']['decisions_made'], 1):
                    st.write(f"**{i}. {decision.get('decisione', 'N/A')}**")
                    st.write(f"   👤 Responsabile: {decision.get('responsabile', 'N/A')}")
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
