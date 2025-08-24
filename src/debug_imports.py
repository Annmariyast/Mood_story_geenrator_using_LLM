import streamlit as st
import time

st.title("🔍 Debug Import Test")

st.write("Testing imports step by step...")

# Test 1: Basic imports
try:
    st.write("✅ Streamlit imported successfully")
except Exception as e:
    st.error(f"❌ Streamlit import failed: {e}")

# Test 2: Path setup
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    st.write("✅ Path setup successful")
except Exception as e:
    st.error(f"❌ Path setup failed: {e}")

# Test 3: Torch import
try:
    import torch
    st.write(f"✅ PyTorch imported successfully: {torch.__version__}")
    st.write(f"✅ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    st.write(f"❌ PyTorch import failed: {e}")

# Test 4: Transformers import
try:
    from transformers import pipeline
    st.write("✅ Transformers imported successfully")
except Exception as e:
    st.write(f"❌ Transformers import failed: {e}")

# Test 5: Component imports
try:
    from components.mood_detector import MoodDetector
    st.write("✅ MoodDetector imported successfully")
except Exception as e:
    st.write(f"❌ MoodDetector import failed: {e}")

try:
    from components.story_generator import StoryGenerator
    st.write("✅ StoryGenerator imported successfully")
except Exception as e:
    st.write(f"❌ StoryGenerator import failed: {e}")

try:
    from components.poster_generator import PosterGenerator
    st.write("✅ PosterGenerator imported successfully")
except Exception as e:
    st.write(f"❌ PosterGenerator import failed: {e}")

try:
    from utils.helpers import get_soundtrack_recommendation, format_script, get_genre_icons
    st.write("✅ Helpers imported successfully")
except Exception as e:
    st.write(f"❌ Helpers import failed: {e}")

st.write("---")
st.write("**Summary:** Check which imports are failing above.")
