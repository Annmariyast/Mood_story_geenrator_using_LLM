import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Handle torch import gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Some features will use simplified alternatives.")

# Import components
from components.mood_detector import MoodDetector
from components.story_generator import StoryGenerator
from components.poster_generator import PosterGenerator
from utils.helpers import get_soundtrack_recommendation, format_script, get_genre_icons

# Page configuration
st.set_page_config(
    page_title="Mood to Movie Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .mood-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .script-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        font-family: 'Courier New', monospace;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .poster-container {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(45deg, #f0f0f0, #ffffff);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metrics-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stProgress .st-bo {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Initialize all models with caching"""
    with st.spinner("🤖 Loading AI models... This may take a moment on first run..."):
        try:
            # Check GPU availability
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            st.sidebar.info(f"🔧 Using device: {device.upper()}")
            
            # Initialize components
            mood_detector = MoodDetector()
            story_generator = StoryGenerator()
            poster_generator = PosterGenerator()
            
            st.success("✅ All models loaded successfully!")
            return mood_detector, story_generator, poster_generator
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return None, None, None

def show_model_info():
    """Display information about loaded models"""
    with st.sidebar.expander("🤖 Model Information"):
        st.write("**Mood Detection:**")
        st.write("- j-hartmann/emotion-english-distilroberta-base")
        st.write("- cardiffnlp/twitter-roberta-base-sentiment")
        
        st.write("**Story Generation:**")
        st.write("- microsoft/DialoGPT-medium")
        st.write("- facebook/bart-large")
        
        st.write("**Text Processing:**")
        st.write("- sentence-transformers/all-MiniLM-L6-v2")

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Mood to Movie Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Transform your emotions into cinematic experiences using AI! ✨")
    st.markdown("*Powered by Local LLM models - No API keys needed!*")
    
    # Initialize models
    mood_detector, story_generator, poster_generator = initialize_models()
    
    if not all([mood_detector, story_generator, poster_generator]):
        st.error("Failed to initialize models. Please check your installation.")
        st.stop()
    
    # Show model info
    show_model_info()
    
    # Sidebar for input
    with st.sidebar:
        st.header("🎭 Express Your Mood")
        
        input_method = st.radio(
            "Choose input method:",
            ["Text", "Emoji", "Voice (Text Input)"],
            help="Select how you'd like to express your current mood"
        )
        
        mood_input = ""
        if input_method == "Text":
            mood_input = st.text_area(
                "Describe your current mood:",
                placeholder="e.g., I feel anxious but excited about tomorrow, or I'm nostalgic and thoughtful...",
                height=100,
                help="Be as descriptive as possible for better results!"
            )
        elif input_method == "Emoji":
            mood_input = st.text_input(
                "Express with emojis:",
                placeholder="😊😢🤔💪🌟",
                help="Use multiple emojis to express complex emotions"
            )
        else:
            mood_input = st.text_area(
                "Transcribed speech (paste here):",
                placeholder="Paste your voice-to-text here...",
                height=100
            )
        
        # Quick mood suggestions
        with st.expander("💡 Need inspiration?"):
            mood_examples = {
                "😊 Happy": "I'm feeling incredibly joyful and optimistic today!",
                "😔 Melancholic": "I'm feeling nostalgic and a bit melancholy about the past",
                "😤 Frustrated": "I'm angry about injustice but determined to make a change",
                "🤔 Contemplative": "I'm in a thoughtful, philosophical mood about life",
                "❤️ Romantic": "I'm feeling loving and dreamy about relationships",
                "😨 Anxious": "I'm nervous but excited about upcoming challenges"
            }
            
            for emoji_mood, example in mood_examples.items():
                if st.button(emoji_mood, key=f"example_{emoji_mood}"):
                    mood_input = example
                    st.rerun()
        
        # Advanced options
        with st.expander("🎛️ Advanced Options"):
            intensity = st.slider(
                "Emotion Intensity", 
                1, 10, 5,
                help="How intense is your current emotional state?"
            )
            
            genre_preference = st.selectbox(
                "Preferred Genre",
                ["Auto-detect", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Adventure", "Fantasy"],
                help="Override automatic genre detection"
            )
            
            script_length = st.selectbox(
                "Script Length",
                ["Short (3-4 scenes)", "Medium (5-6 scenes)", "Long (7-8 scenes)"],
                help="How long should your movie script be?"
            )
            
            creative_mode = st.checkbox(
                "🎨 Creative Mode",
                value=False,
                help="Enable more creative and experimental story generation"
            )
        
        st.markdown("---")
        generate_button = st.button("🎬 Generate Movie", type="primary", use_container_width=True)
    
    # Main content area
    if generate_button and mood_input:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Mood Analysis
        status_text.text("🔮 Analyzing your mood...")
        progress_bar.progress(25)
        
        with st.spinner("Analyzing emotional patterns..."):
            mood_analysis = mood_detector.analyze_mood(
                mood_input, 
                input_method.lower(),
                intensity
            )
        
        # Display mood analysis
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="mood-card">', unsafe_allow_html=True)
            st.subheader("🎭 Mood Analysis")
            
            # Get genre icon
            icons = get_genre_icons()
            genre_icon = icons.get(mood_analysis['suggested_genre'], '🎬')
            
            st.write(f"**Primary Emotion:** {mood_analysis['primary_emotion'].title()} {mood_analysis.get('emotion_emoji', '😊')}")
            st.write(f"**Intensity:** {mood_analysis['intensity']}/10")
            st.write(f"**Suggested Genre:** {genre_icon} {mood_analysis['suggested_genre']}")
            st.write(f"**Confidence:** {mood_analysis['confidence']:.1%}")
            
            # Show additional emotions
            if len(mood_analysis.get('all_emotions', [])) > 1:
                st.write("**Other Emotions:**")
                for emotion in mood_analysis['all_emotions'][1:3]:  # Show top 2 additional
                    st.write(f"- {emotion['label'].title()}: {emotion['score']:.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Step 2: Story Generation
            status_text.text("✍️ Crafting your personalized story...")
            progress_bar.progress(50)
            
            with st.spinner("Generating story elements..."):
                story_data = story_generator.generate_story(
                    mood_analysis,
                    genre_preference if genre_preference != "Auto-detect" else None,
                    script_length,
                    creative_mode
                )
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📖 Your Movie")
            st.markdown(f"**🎬 Title:** {story_data['title']}")
            st.markdown(f"**✨ Tagline:** *{story_data['tagline']}*")
            st.markdown(f"**📝 Summary:** {story_data['summary']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 3: Poster Generation
        status_text.text("🎨 Creating your movie poster...")
        progress_bar.progress(75)
        
        st.subheader("🎨 Movie Poster")
        with st.spinner("Generating artistic poster..."):
            poster_result = poster_generator.generate_poster(
                story_data['title'],
                story_data['summary'],
                mood_analysis['suggested_genre'],
                style="cinematic"
            )
        
        if poster_result and poster_result.get('image'):
            st.markdown('<div class="poster-container">', unsafe_allow_html=True)
            st.image(
                poster_result['image'], 
                width=400, 
                caption=f"🎬 Poster for '{story_data['title']}'"
            )
            if poster_result.get('description'):
                st.caption(f"🎨 {poster_result['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 4: Complete
        status_text.text("🎉 Your movie is ready!")
        progress_bar.progress(100)
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Display script
        st.subheader("📝 Movie Script")
        formatted_script = format_script(story_data['script'])
        st.markdown(f'<div class="script-container">{formatted_script}</div>', unsafe_allow_html=True)
        
        # Story metrics
        metrics = story_generator.get_story_metrics(story_data['script'])
        if metrics:
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            st.subheader("📊 Script Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Word Count", metrics.get('word_count', 0))
            with col2:
                st.metric("Scenes", metrics.get('estimated_scenes', 0))
            with col3:
                st.metric("Reading Time", f"{metrics.get('reading_time_minutes', 0):.1f} min")
            with col4:
                st.metric("Est. Runtime", f"{metrics.get('estimated_screen_time_minutes', 0):.1f} min")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Soundtrack recommendation
        st.subheader("🎵 Soundtrack Recommendation")
        soundtrack = get_soundtrack_recommendation(mood_analysis['primary_emotion'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**🎼 Recommended Genre:** {soundtrack['genre']}")
            st.write(f"**🎭 Mood:** {soundtrack['mood']}")
            st.write(f"**🎹 Key Elements:** {', '.join(soundtrack['elements'])}")
        with col2:
            st.write(f"**🎤 Sample Artists:** {', '.join(soundtrack['artists'])}")
            st.write("**🎵 Sample Tracks:**")
            for track in soundtrack.get('sample_tracks', [])[:3]:
                st.write(f"• {track}")
        
        # Multiple endings option
        with st.expander("🔄 Alternative Endings"):
            if st.button("Generate Alternative Endings", key="alt_endings"):
                with st.spinner("Creating alternative storylines..."):
                    alt_endings = story_generator.generate_alternative_endings(story_data, count=3)
                    for i, ending in enumerate(alt_endings, 1):
                        st.write(f"**Ending {i}:** {ending}")
        
        # Download options
        st.subheader("📥 Download Your Movie Kit")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            script_content = f"""MOVIE SCRIPT: {story_data['title']}
{'-'*50}

TAGLINE: {story_data['tagline']}
GENRE: {mood_analysis['suggested_genre']}

SUMMARY:
{story_data['summary']}

{'-'*50}
SCRIPT:
{'-'*50}

{story_data['script']}

{'-'*50}
Generated by Mood to Movie Generator
Emotion: {mood_analysis['primary_emotion']} (Intensity: {mood_analysis['intensity']}/10)
"""
            st.download_button(
                label="📄 Download Script",
                data=script_content,
                file_name=f"{story_data['title'].replace(' ', '_')}_script.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if poster_result and poster_result.get('image_bytes'):
                st.download_button(
                    label="🎨 Download Poster",
                    data=poster_result['image_bytes'],
                    file_name=f"{story_data['title'].replace(' ', '_')}_poster.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("Poster download not available")
        
        with col3:
            summary_content = f"""🎬 MY AI MOVIE 🎬

{genre_icon} {story_data['title']}
✨ {story_data['tagline']}

🎭 Genre: {mood_analysis['suggested_genre']}
😊 Mood: {mood_analysis['primary_emotion']} ({mood_analysis['intensity']}/10)

📖 {story_data['summary']}

🎵 Soundtrack: {soundtrack['genre']}

Generated with Mood to Movie Generator ✨
#MoodToMovie #AIStorytelling #LocalLLM"""
            
            st.download_button(
                label="📱 Share Summary",
                data=summary_content,
                file_name=f"{story_data['title'].replace(' ', '_')}_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    elif generate_button and not mood_input:
        st.error("🚫 Please enter your mood first!")
        st.info("💡 Try describing how you feel or use emojis to express your emotions!")
    
    # Footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🤖 **Powered by Local AI**")
        st.caption("Transformers • PyTorch • Hugging Face")
    with col2:
        st.markdown("🔐 **Privacy First**")
        st.caption("All processing happens locally")
    with col3:
        st.markdown("⚡ **No API Keys**")
        st.caption("100% open source models")

if __name__ == "__main__":
    main()