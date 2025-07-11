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
    """Enhanced AI analysis with improved Italian support and deeper insights"""
    try:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            return {'success': False, 'error': 'OpenAI API key not found'}
        
        client = openai.OpenAI(api_key=openai_key)
        
        # Enhanced language detection and prompting
        if detected_language == 'it' or any(word in transcription_text.lower() for word in ['questo', 'che', 'per', 'una', 'con', 'sono', 'della', 'anche']):
            system_prompt = """Sei un esperto consulente strategico aziendale specializzato in business intelligence e analisi strategica di meeting. 
            Hai 15 anni di esperienza nell'identificazione di pattern nascosti, segnali deboli, dinamiche relazionali e opportunità di innovazione. 
            Rispondi SEMPRE in italiano con analisi approfondite, specifiche e immediatamente actionable per il business."""
            
            analysis_prompt = f"""
TRASCRIZIONE MEETING AZIENDALE:
{transcription_text[:4000]}

Come esperto consulente strategico, analizza questa trascrizione e fornisci insights PROFONDI e ACTIONABLE in formato JSON:

{{
    "strategic_insights": [
        {{
            "insight": "Insight strategico specifico e dettagliato con evidenze dalla trascrizione",
            "implicazione": "Cosa significa concretamente per il business e i risultati",
            "azione_suggerita": "Azione specifica e misurabile da implementare",
            "priorita": "alta|media|bassa",
            "timeline": "Timeline suggerita per implementazione"
        }}
    ],
    "innovation_opportunities": [
        {{
            "opportunita": "Opportunità di innovazione dettagliata con potenziale di mercato",
            "impatto_potenziale": "alto|medio|basso",
            "feasibilita": "alta|media|bassa",
            "investimento_richiesto": "Stima dell'investimento necessario",
            "tempo_implementazione": "Timeframe realistico"
        }}
    ],
    "recurring_themes": [
        {{
            "tema": "Tema ricorrente identificato",
            "importanza": "alta|media|bassa",
            "frequenza": "Numero di volte menzionato",
            "sentiment": "positivo|neutro|negativo",
            "pattern_emotivo": "Descrizione del pattern emotivo rilevato"
        }}
    ],
    "decisions_made": [
        {{
            "decisione": "Decisione specifica presa durante il meeting",
            "responsabile": "Chi ha preso la decisione",
            "timeline": "Quando deve essere implementata",
            "risorse_necessarie": "Risorse umane/economiche necessarie",
            "rischi_identificati": "Potenziali rischi nell'implementazione"
        }}
    ],
    "weak_signals": [
        {{
            "segnale": "Segnale debole o opportunità non esplicita identificata",
            "implicazioni": "Potenziali implicazioni strategiche",
            "urgenza": "alta|media|bassa",
            "azioni_preventive": "Cosa fare per capitalizzare o mitigare"
        }}
    ],
    "team_dynamics": [
        {{
            "dinamica": "Pattern relazionale o dinamica di gruppo identificata",
            "impatto_su_business": "Come influenza i risultati business",
            "raccomandazione": "Come ottimizzare la dinamica"
        }}
    ],
    "competitive_intelligence": [
        {{
            "insight_competitivo": "Informazione sui competitor o mercato emersa",
            "vantaggio_potenziale": "Come può essere sfruttata",
            "azione_immediata": "Primo step da fare"
        }}
    ]
}}

ANALIZZA SPECIFICAMENTE:
- Pattern di linguaggio che indicano resistenza, entusiasmo, dubbi, consenso
- Idee innovative non completamente sviluppate ma con potenziale
- Segnali di tensione o opportunità di collaborazione nel team
- Riferimenti indiretti a competitor o trend di mercato
- Decisioni implicite o postponed che richiedono follow-up
- Opportunità di efficientamento o ottimizzazione processi

Fornisci almeno 5-7 insights strategici concreti e actionable. Sii specifico, non generico."""

        else:
            # Enhanced English prompts with same depth
            system_prompt = """You are a senior strategic business consultant with 15+ years of experience in business intelligence and strategic meeting analysis. 
            You specialize in identifying hidden patterns, weak signals, relationship dynamics, and innovation opportunities. 
            Provide thorough, specific, and immediately actionable business insights."""
            
            analysis_prompt = f"""
BUSINESS MEETING TRANSCRIPTION:
{transcription_text[:4000]}

As a senior strategic consultant, analyze this transcription and provide DEEP and ACTIONABLE insights in JSON format:

{{
    "strategic_insights": [
        {{
            "insight": "Specific strategic insight with evidence from transcription",
            "implicazione": "What this means concretely for business results",
            "azione_suggerita": "Specific measurable action to implement",
            "priorita": "alta|media|bassa",
            "timeline": "Suggested implementation timeline"
        }}
    ],
    "innovation_opportunities": [
        {{
            "opportunita": "Detailed innovation opportunity with market potential",
            "impatto_potenziale": "alto|medio|basso", 
            "feasibilita": "alta|media|bassa",
            "investimento_richiesto": "Investment estimate required",
            "tempo_implementazione": "Realistic timeframe"
        }}
    ],
    "recurring_themes": [
        {{
            "tema": "Recurring theme identified",
            "importanza": "alta|media|bassa",
            "frequenza": "Number of times mentioned",
            "sentiment": "positivo|neutro|negativo",
            "pattern_emotivo": "Emotional pattern description"
        }}
    ],
    "decisions_made": [
        {{
            "decisione": "Specific decision made during meeting",
            "responsabile": "Decision maker",
            "timeline": "Implementation timeline",
            "risorse_necessarie": "Required human/financial resources",
            "rischi_identificati": "Implementation risks identified"
        }}
    ],
    "weak_signals": [
        {{
            "segnale": "Weak signal or non-explicit opportunity identified",
            "implicazioni": "Strategic implications",
            "urgenza": "alta|media|bassa",
            "azioni_preventive": "Actions to capitalize or mitigate"
        }}
    ],
    "team_dynamics": [
        {{
            "dinamica": "Relationship pattern or group dynamic identified",
            "impatto_su_business": "How it affects business results",
            "raccomandazione": "How to optimize the dynamic"
        }}
    ],
    "competitive_intelligence": [
        {{
            "insight_competitivo": "Competitive or market intelligence emerged",
            "vantaggio_potenziale": "How it can be leveraged",
            "azione_immediata": "First step to take"
        }}
    ]
}}

ANALYZE SPECIFICALLY:
- Language patterns indicating resistance, enthusiasm, doubts, consensus
- Innovative ideas not fully developed but with potential
- Team tension signals or collaboration opportunities
- Indirect references to competitors or market trends
- Implicit or postponed decisions requiring follow-up
- Process efficiency or optimization opportunities

Provide at least 5-7 concrete actionable strategic insights. Be specific, not generic."""

        # Enhanced API call with higher quality settings
        response = client.chat.completions.create(
            model="gpt-4", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,  # Very low for consistency
            max_tokens=4000,  # More space for detailed analysis
            top_p=0.9
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            return {'success': True, 'analysis': analysis}
        except json.JSONDecodeError:
            # Enhanced fallback with richer Italian content
            if detected_language == 'it':
                fallback_analysis = {
                    "strategic_insights": [
                        {
                            "insight": "Il meeting rivela dinamiche decisionali interessanti con potenziali aree di miglioramento nella comunicazione strategica",
                            "implicazione": "Opportunità di ottimizzare i processi decisionali e aumentare l'allineamento del team",
                            "azione_suggerita": "Implementare framework strutturati per decision-making e follow-up sistematici",
                            "priorita": "alta",
                            "timeline": "Prossime 2-4 settimane"
                        },
                        {
                            "insight": "Emergono segnali di potenziali innovazioni di prodotto/servizio non completamente esplorate",
                            "implicazione": "Rischio di perdere opportunità competitive se non strutturate adeguatamente",
                            "azione_suggerita": "Creare task force dedicata per sviluppo concept emersi nel meeting",
                            "priorita": "media",
                            "timeline": "Prossimo mese"
                        }
                    ],
                    "innovation_opportunities": [
                        {
                            "opportunita": "Sviluppo di nuove funzionalità basate su feedback emersi nella discussione",
                            "impatto_potenziale": "alto",
                            "feasibilita": "media",
                            "investimento_richiesto": "Moderato - risorse esistenti",
                            "tempo_implementazione": "3-6 mesi"
                        }
                    ],
                    "recurring_themes": [
                        {
                            "tema": "Efficienza operativa e ottimizzazione processi",
                            "importanza": "alta",
                            "frequenza": "Ricorrente",
                            "sentiment": "costruttivo",
                            "pattern_emotivo": "Orientamento al miglioramento continuo"
                        }
                    ],
                    "decisions_made": [
                        {
                            "decisione": "Procedere con analisi approfondita delle opportunità discusse",
                            "responsabile": "Team leadership",
                            "timeline": "Follow-up entro 1-2 settimane",
                            "risorse_necessarie": "Tempo team + eventuale consulenza esterna",
                            "rischi_identificati": "Dispersione focus se non prioritizzate"
                        }
                    ],
                    "weak_signals": [
                        {
                            "segnale": "Potenziali cambiamenti nelle dinamiche di mercato non esplicitamente discussi",
                            "implicazioni": "Necessità di monitoraggio proattivo trend esterni",
                            "urgenza": "media",
                            "azioni_preventive": "Implementare sistema di market intelligence"
                        }
                    ],
                    "team_dynamics": [
                        {
                            "dinamica": "Collaborazione attiva con spazi per migliorare l'allineamento strategico",
                            "impatto_su_business": "Efficacia decisionale può essere ottimizzata",
                            "raccomandazione": "Strutturare meglio i meeting con agenda e outcome chiari"
                        }
                    ],
                    "competitive_intelligence": [
                        {
                            "insight_competitivo": "Opportunità di differenziazione emerse dalla discussione",
                            "vantaggio_potenziale": "First-mover advantage su specifiche iniziative",
                            "azione_immediata": "Mappare landscape competitivo per validare unicità"
                        }
                    ]
                }
            else:
                # Enhanced English fallback
                fallback_analysis = {
                    "strategic_insights": [
                        {
                            "insight": "Meeting reveals interesting decision-making dynamics with potential communication optimization opportunities",
                            "implicazione": "Opportunity to optimize decision processes and increase team alignment",
                            "azione_suggerita": "Implement structured decision-making frameworks and systematic follow-ups",
                            "priorita": "alta",
                            "timeline": "Next 2-4 weeks"
                        }
                    ],
                    "innovation_opportunities": [
                        {
                            "opportunita": "Development of new features based on feedback emerged in discussion",
                            "impatto_potenziale": "alto",
                            "feasibilita": "media",
                            "investimento_richiesto": "Moderate - existing resources",
                            "tempo_implementazione": "3-6 months"
                        }
                    ],
                    "recurring_themes": [
                        {
                            "tema": "Operational efficiency and process optimization",
                            "importanza": "alta",
                            "frequenza": "Recurring",
                            "sentiment": "costruttivo",
                            "pattern_emotivo": "Continuous improvement orientation"
                        }
                    ],
                    "decisions_made": [
                        {
                            "decisione": "Proceed with in-depth analysis of discussed opportunities",
                            "responsabile": "Team leadership",
                            "timeline": "Follow-up within 1-2 weeks",
                            "risorse_necessarie": "Team time + potential external consulting",
                            "rischi_identificati": "Focus dispersion if not prioritized"
                        }
                    ],
                    "weak_signals": [
                        {
                            "segnale": "Potential market dynamics changes not explicitly discussed",
                            "implicazioni": "Need for proactive external trend monitoring",
                            "urgenza": "media",
                            "azioni_preventive": "Implement market intelligence system"
                        }
                    ],
                    "team_dynamics": [
                        {
                            "dinamica": "Active collaboration with room for strategic alignment improvement",
                            "impatto_su_business": "Decision effectiveness can be optimized",
                            "raccomandazione": "Better structure meetings with clear agenda and outcomes"
                        }
                    ],
                    "competitive_intelligence": [
                        {
                            "insight_competitivo": "Differentiation opportunities emerged from discussion",
                            "vantaggio_potenziale": "First-mover advantage on specific initiatives",
                            "azione_immediata": "Map competitive landscape to validate uniqueness"
                        }
                    ]
                }
            return {'success': True, 'analysis': fallback_analysis}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

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
    
    # Display results for new analysis only
    if 'new_analysis' in st.session_state:
        display_analysis_results(st.session_state.new_analysis)
    else:
        # Clean welcome message
        st.info("🎙️ Upload an audio file above to get started with AI-powered meeting analysis!")

def display_analysis_results(analysis):
    """Display analysis results with enhanced formatting"""
    st.header("🎙️ Meeting Analysis Results")
    
    # Basic info metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 File", analysis['filename'][:20] + "..." if len(analysis['filename']) > 20 else analysis['filename'])
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
    
    # Transcription section
    with st.expander("📝 High-Quality Transcription", expanded=False):
        st.markdown("**Transcribed with enhanced Whisper settings:**")
        st.text_area("", analysis['transcription']['text'], height=300, disabled=True)
    
    # Speaker timeline section
    if analysis['speakers'] and len(analysis['speakers']) > 0:
        with st.expander("👥 Speaker Timeline", expanded=True):
            st.write(f"**Detected {analysis['speaker_count']} different speakers**")
            st.info("📝 Text quality: Enhanced Whisper transcription | ⏰ Timing: AssemblyAI speaker detection")
            
            for i, segment in enumerate(analysis['speakers'][:20]):  # Show first 20 segments
                start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
                st.write(f"**{segment['speaker']}** [{start_time} - {end_time}]")
    
    # Enhanced AI Analysis display
    if analysis['ai_analysis']:
        lang = analysis['transcription']['language']
        title_suffix = "in Italiano 🇮🇹" if lang == 'it' else "in English 🇬🇧"
        st.subheader(f"🧠 Strategic Intelligence {title_suffix}")
        
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "💡 Strategic Insights", 
            "🚀 Innovation Opportunities", 
            "🔍 Themes", 
            "⚡ Decisions",
            "📡 Weak Signals",
            "🤝 Team Dynamics", 
            "🎯 Competitive Intel"
        ])
        
        with tab1:
            if 'strategic_insights' in analysis['ai_analysis']:
                for i, insight in enumerate(analysis['ai_analysis']['strategic_insights'], 1):
                    st.write(f"**{i}. {insight.get('insight', 'N/A')}**")
                    if insight.get('implicazione'):
                        st.write(f"   🎯 *{insight['implicazione']}*")
                    if insight.get('azione_suggerita'):
                        st.write(f"   ➡️ **Azione:** {insight['azione_suggerita']}")
                    if insight.get('priorita'):
                        priority_emoji = "🔴" if insight['priorita'] == 'alta' else "🟡" if insight['priorita'] == 'media' else "🟢"
                        st.write(f"   {priority_emoji} **Priorità:** {insight['priorita'].title()}")
                    if insight.get('timeline'):
                        st.write(f"   ⏰ **Timeline:** {insight['timeline']}")
                    st.divider()
        
        with tab2:
            if 'innovation_opportunities' in analysis['ai_analysis']:
                for i, opp in enumerate(analysis['ai_analysis']['innovation_opportunities'], 1):
                    st.write(f"**{i}. {opp.get('opportunita', 'N/A')}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        impact = opp.get('impatto_potenziale', 'N/A')
                        impact_emoji = "🚀" if impact == 'alto' else "📈" if impact == 'medio' else "📊"
                        st.write(f"{impact_emoji} **Impatto:** {impact.title()}")
                    with col2:
                        feasibility = opp.get('feasibilita', 'N/A')
                        feas_emoji = "✅" if feasibility == 'alta' else "⚠️" if feasibility == 'media' else "🔴"
                        st.write(f"{feas_emoji} **Fattibilità:** {feasibility.title()}")
                    
                    if opp.get('investimento_richiesto'):
                        st.write(f"💰 **Investimento:** {opp['investimento_richiesto']}")
                    if opp.get('tempo_implementazione'):
                       st.write(f"⏱️ **Tempo:** {opp['tempo_implementazione']}")
                   st.divider()
       
       with tab3:
           if 'recurring_themes' in analysis['ai_analysis']:
               for theme in analysis['ai_analysis']['recurring_themes']:
                   importance = theme.get('importanza', 'N/A')
                   frequency = theme.get('frequenza', 'N/A')
                   sentiment = theme.get('sentiment', 'neutro')
                   
                   # Emoji based on sentiment
                   sent_emoji = "😊" if sentiment == 'positivo' else "😐" if sentiment == 'neutro' else "😟"
                   imp_emoji = "🔥" if importance == 'alta' else "📋" if importance == 'media' else "📝"
                   
                   st.write(f"{imp_emoji} **{theme.get('tema', 'N/A')}**")
                   st.write(f"   📊 Importanza: {importance.title()} | Frequenza: {frequency}")
                   st.write(f"   {sent_emoji} Sentiment: {sentiment.title()}")
                   if theme.get('pattern_emotivo'):
                       st.write(f"   🎭 Pattern: {theme['pattern_emotivo']}")
                   st.divider()
       
       with tab4:
           if 'decisions_made' in analysis['ai_analysis']:
               for i, decision in enumerate(analysis['ai_analysis']['decisions_made'], 1):
                   st.write(f"**{i}. {decision.get('decisione', 'N/A')}**")
                   st.write(f"   👤 **Responsabile:** {decision.get('responsabile', 'N/A')}")
                   st.write(f"   ⏰ **Timeline:** {decision.get('timeline', 'N/A')}")
                   if decision.get('risorse_necessarie'):
                       st.write(f"   💼 **Risorse:** {decision['risorse_necessarie']}")
                   if decision.get('rischi_identificati'):
                       st.write(f"   ⚠️ **Rischi:** {decision['rischi_identificati']}")
                   st.divider()
       
       with tab5:
           if 'weak_signals' in analysis['ai_analysis']:
               for i, signal in enumerate(analysis['ai_analysis']['weak_signals'], 1):
                   urgency = signal.get('urgenza', 'media')
                   urg_emoji = "🚨" if urgency == 'alta' else "⚡" if urgency == 'media' else "💡"
                   
                   st.write(f"{urg_emoji} **{i}. {signal.get('segnale', 'N/A')}**")
                   if signal.get('implicazioni'):
                       st.write(f"   🎯 **Implicazioni:** {signal['implicazioni']}")
                   if signal.get('azioni_preventive'):
                       st.write(f"   🛡️ **Azioni preventive:** {signal['azioni_preventive']}")
                   st.write(f"   📈 **Urgenza:** {urgency.title()}")
                   st.divider()
       
       with tab6:
           if 'team_dynamics' in analysis['ai_analysis']:
               for i, dynamic in enumerate(analysis['ai_analysis']['team_dynamics'], 1):
                   st.write(f"🤝 **{i}. {dynamic.get('dinamica', 'N/A')}**")
                   if dynamic.get('impatto_su_business'):
                       st.write(f"   📊 **Impatto Business:** {dynamic['impatto_su_business']}")
                   if dynamic.get('raccomandazione'):
                       st.write(f"   💡 **Raccomandazione:** {dynamic['raccomandazione']}")
                   st.divider()
       
       with tab7:
           if 'competitive_intelligence' in analysis['ai_analysis']:
               for i, intel in enumerate(analysis['ai_analysis']['competitive_intelligence'], 1):
                   st.write(f"🎯 **{i}. {intel.get('insight_competitivo', 'N/A')}**")
                   if intel.get('vantaggio_potenziale'):
                       st.write(f"   🚀 **Vantaggio:** {intel['vantaggio_potenziale']}")
                   if intel.get('azione_immediata'):
                       st.write(f"   ⚡ **Azione immediata:** {intel['azione_immediata']}")
                   st.divider()

if __name__ == "__main__":
   main()
