import streamlit as st
import re
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Handle torch import with fallback
try:
    import torch
    TORCH_AVAILABLE = True
    st.info("✅ PyTorch loaded successfully")
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Using simplified mood detection.")

# Handle emoji import
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    st.warning("⚠️ Emoji package not available. Some features may be limited.")

# Handle transformers import
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Using rule-based mood detection.")

class MoodDetector:
    """Advanced mood detection with fallback to rule-based detection"""
    
    def __init__(self):
        self.device = 0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
        
        if TRANSFORMERS_AVAILABLE:
            self.emotion_classifier = self._load_emotion_classifier()
            self.sentiment_classifier = self._load_sentiment_classifier()
        else:
            self.emotion_classifier = None
            self.sentiment_classifier = None
            
        self.emoji_to_emotion = self._create_emoji_mapping()
        self.emotion_to_genre = self._create_emotion_genre_mapping()
        self.intensity_keywords = self._create_intensity_keywords()
        
    @st.cache_resource
    def _load_emotion_classifier(_self):
        """Load emotion classification model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=_self.device,
                return_all_scores=True
            )
            return classifier
        except Exception as e:
            st.warning(f"Primary emotion model unavailable, using fallback: {e}")
            try:
                # Fallback to lighter model
                classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-emotion",
                    device=_self.device,
                    return_all_scores=True
                )
                return classifier
            except:
                return None
    
    @st.cache_resource  
    def _load_sentiment_classifier(_self):
        """Load sentiment classification model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=_self.device,
                return_all_scores=True
            )
            return classifier
        except Exception as e:
            st.warning(f"Sentiment model unavailable: {e}")
            return None
    
    def detect_mood(self, text: str) -> Dict[str, Any]:
        """Detect mood from text with fallback to rule-based detection"""
        if self.emotion_classifier and self.sentiment_classifier:
            return self._ai_mood_detection(text)
        else:
            return self._rule_based_mood_detection(text)
    
    def _rule_based_mood_detection(self, text: str) -> Dict[str, Any]:
        """Rule-based mood detection when AI models aren't available"""
        text_lower = text.lower()
        
        # Simple keyword-based mood detection
        mood_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic'],
            'sad': ['sad', 'depressed', 'miserable', 'unhappy', 'gloomy', 'melancholy'],
            'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'quiet'],
            'excited': ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped'],
            'nervous': ['nervous', 'anxious', 'worried', 'stressed', 'tense'],
            'romantic': ['romantic', 'loving', 'passionate', 'tender', 'affectionate']
        }
        
        detected_moods = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                detected_moods[mood] = score / len(keywords)
        
        # Get the dominant mood
        if detected_moods:
            dominant_mood = max(detected_moods, key=detected_moods.get)
            confidence = detected_moods[dominant_mood]
        else:
            dominant_mood = "neutral"
            confidence = 0.5
        
        return {
            'primary_mood': dominant_mood,
            'confidence': confidence,
            'mood_scores': detected_moods,
            'method': 'rule_based'
        }
    
    def _ai_mood_detection(self, text: str) -> Dict[str, Any]:
        """AI-based mood detection when models are available"""
        # Initialize results
        emotions = []
        sentiment_info = {}
        intensity_score = 5 # Default base intensity
        
        # Analyze based on input type
        if TRANSFORMERS_AVAILABLE:
            emotions = self._analyze_text_emotion(text)
            sentiment_info = self._analyze_sentiment(text)
            intensity_score = self._calculate_text_intensity(text, 5)
            
        elif EMOJI_AVAILABLE:
            emotions = self._analyze_emoji_emotion(text)
            intensity_score = self._calculate_emoji_intensity(text, 5)
        
        # Fallback if no emotions detected
        if not emotions:
            emotions = [{'label': 'neutral', 'score': 0.5}]
        
        primary_emotion = emotions[0]['label']
        confidence = emotions[0]['score']
        
        # Get genre suggestion
        suggested_genre = self._get_genre_from_emotion(primary_emotion)
        
        # Add emotion emoji
        emotion_emoji = self._get_emotion_emoji(primary_emotion)
        
        # Calculate final intensity
        final_intensity = min(10, max(1, int(intensity_score)))
        
        # Get mood insights
        insights = self._get_mood_insights(primary_emotion, final_intensity)
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'intensity': final_intensity,
            'suggested_genre': suggested_genre,
            'all_emotions': emotions[:5],  # Top 5 emotions
            'sentiment_info': sentiment_info,
            'raw_input': text,
            'input_type': 'text', # Assuming text input for AI detection
            'emotion_emoji': emotion_emoji,
            'insights': insights,
            'mood_description': insights.get('description', ''),
            'recommended_themes': insights.get('themes', []),
            'narrative_tone': insights.get('tone', 'balanced')
        }
    
    def _analyze_text_emotion(self, text: str) -> List[Dict[str, Any]]:
        """Analyze emotion from text using transformer models"""
        if not self.emotion_classifier or not text.strip():
            return self._fallback_text_emotion(text)
        
        try:
            # Clean text
            clean_text = self._clean_text_for_analysis(text)
            
            # Get predictions
            results = self.emotion_classifier(clean_text)
            
            # Sort by confidence
            if isinstance(results[0], list):
                results = results[0]
            
            return sorted(results, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            st.warning(f"Error in emotion analysis: {e}")
            return self._fallback_text_emotion(text)
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment to complement emotion detection"""
        if not self.sentiment_classifier or not text.strip():
            return {}
        
        try:
            results = self.sentiment_classifier(text)
            if isinstance(results[0], list):
                results = results[0]
            
            return {
                'sentiment': results[0]['label'],
                'sentiment_score': results[0]['score'],
                'all_sentiments': results
            }
            
        except Exception as e:
            return {}
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """Clean text for better analysis"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle emojis in text (keep them)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text
    
    def _calculate_text_intensity(self, text: str, base_intensity: int) -> float:
        """Calculate emotional intensity from text"""
        text_lower = text.lower()
        intensity_modifier = 0
        
        # Check for intensity keywords
        for level, keywords in self.intensity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'high':
                        intensity_modifier += 2
                    elif level == 'medium':
                        intensity_modifier += 1
                    elif level == 'low':
                        intensity_modifier -= 1
        
        # Check for punctuation intensity
        exclamation_count = text.count('!')
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        intensity_modifier += exclamation_count * 0.5
        intensity_modifier += caps_words * 0.3
        
        # Text length factor (longer descriptions might indicate more intensity)
        word_count = len(text.split())
        if word_count > 50:
            intensity_modifier += 1
        elif word_count > 20:
            intensity_modifier += 0.5
        
        return base_intensity + intensity_modifier
    
    def _analyze_emoji_emotion(self, emoji_text: str) -> List[Dict[str, Any]]:
        """Analyze emotions from emojis"""
        emoji_chars = []
        
        # Extract emojis
        for char in emoji_text:
            if char in emoji.EMOJI_DATA or char in self.emoji_to_emotion:
                emoji_chars.append(char)
        
        # Try regex pattern for additional emojis
        if not emoji_chars:
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+")
            emoji_chars = emoji_pattern.findall(emoji_text)
        
        emotion_counts = {}
        total_emojis = 0
        
        for emoji_char in emoji_chars:
            total_emojis += 1
            if emoji_char in self.emoji_to_emotion:
                emotion = self.emoji_to_emotion[emoji_char]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return []
        
        # Convert to classifier format
        emotions = []
        for emotion, count in emotion_counts.items():
            emotions.append({
                'label': emotion,
                'score': count / total_emojis
            })
        
        return sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    def _calculate_emoji_intensity(self, emoji_text: str, base_intensity: int) -> float:
        """Calculate intensity from emoji usage"""
        emoji_count = len([char for char in emoji_text if char in emoji.EMOJI_DATA])
        
        # High-intensity emojis
        high_intensity_emojis = ['😭', '😡', '🤬', '😱', '🤯', '💔', '🔥', '💥', '⚡']
        low_intensity_emojis = ['😌', '😊', '🙂', '😐', '😑']
        
        intensity_modifier = 0
        
        for char in emoji_text:
            if char in high_intensity_emojis:
                intensity_modifier += 2
            elif char in low_intensity_emojis:
                intensity_modifier -= 0.5
        
        # Multiple same emojis indicate intensity
        for emoji_char in set(emoji_text):
            count = emoji_text.count(emoji_char)
            if count > 2:
                intensity_modifier += count * 0.3
        
        return base_intensity + intensity_modifier
    
    def _fallback_text_emotion(self, text: str) -> List[Dict[str, Any]]:
        """Fallback emotion detection using keywords"""
        emotion_keywords = {
            'joy': ['happy', 'joyful', 'excited', 'elated', 'cheerful', 'delighted', 'thrilled'],
            'sadness': ['sad', 'depressed', 'melancholy', 'down', 'blue', 'gloomy', 'sorrowful'],
            'anger': ['angry', 'furious', 'mad', 'rage', 'frustrated', 'annoyed', 'irritated'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified', 'frightened'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'stunned', 'bewildered'],
            'love': ['love', 'romantic', 'affectionate', 'adore', 'cherish', 'devoted'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened'],
            'contempt': ['contempt', 'disdain', 'scorn', 'disgust']
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        if not emotion_scores:
            return [{'label': 'neutral', 'score': 0.5}]
        
        # Convert to list format
        emotions = [{'label': emotion, 'score': score} for emotion, score in emotion_scores.items()]
        return sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    def _get_genre_from_emotion(self, emotion: str) -> str:
        """Map emotion to movie genre"""
        emotion_lower = emotion.lower()
        
        # Direct mapping
        if emotion_lower in self.emotion_to_genre:
            return self.emotion_to_genre[emotion_lower]
        
        # Pattern matching
        if any(word in emotion_lower for word in ['joy', 'happy', 'excited', 'amused']):
            return 'Comedy'
        elif any(word in emotion_lower for word in ['sad', 'melancholy', 'grief', 'disappointed']):
            return 'Drama'
        elif any(word in emotion_lower for word in ['angry', 'rage', 'frustrated', 'annoyed']):
            return 'Thriller'
        elif any(word in emotion_lower for word in ['scared', 'afraid', 'terrified', 'fear']):
            return 'Horror'
        elif any(word in emotion_lower for word in ['love', 'romantic', 'affection']):
            return 'Romance'
        elif any(word in emotion_lower for word in ['surprised', 'amazed', 'wonder', 'curious']):
            return 'Adventure'
        else:
            return 'Drama'  # Default
    
    def _get_emotion_emoji(self, emotion: str) -> str:
        """Get representative emoji for emotion"""
        emotion_emojis = {
            'joy': '😊',
            'happiness': '😄',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😰',
            'surprise': '😲',
            'love': '❤️',
            'disgust': '🤢',
            'contempt': '😤',
            'excitement': '🤩',
            'anxiety': '😰',
            'peace': '😌',
            'confidence': '😎',
            'contemplation': '🤔',
            'neutral': '😐'
        }
        
        return emotion_emojis.get(emotion.lower(), '😊')
    
    def _get_mood_insights(self, emotion: str, intensity: int) -> Dict[str, Any]:
        """Generate insights about the detected mood"""
        
        # Intensity descriptions
        intensity_desc = {
            1: 'very mild', 2: 'mild', 3: 'light', 4: 'moderate', 5: 'average',
            6: 'strong', 7: 'intense', 8: 'very intense', 9: 'overwhelming', 10: 'extreme'
        }
        
        # Emotion descriptions
        emotion_descriptions = {
            'joy': f"You're experiencing {intensity_desc[intensity]} joy and happiness. This suggests a positive outlook and energy.",
            'sadness': f"You're feeling {intensity_desc[intensity]} sadness or melancholy. This often leads to introspective and meaningful stories.",
            'anger': f"You're experiencing {intensity_desc[intensity]} anger or frustration. This energy can drive powerful, justice-themed narratives.",
            'fear': f"You're feeling {intensity_desc[intensity]} fear or anxiety. This creates tension perfect for suspenseful storytelling.",
            'surprise': f"You're experiencing {intensity_desc[intensity]} surprise or wonder. This opens doors to discovery and adventure.",
            'love': f"You're feeling {intensity_desc[intensity]} love or affection. This warmth translates beautifully into romantic narratives.",
            'excitement': f"You're feeling {intensity_desc[intensity]} excitement. This energy is perfect for dynamic, adventurous stories."
        }
        
        # Recommended themes
        emotion_themes = {
            'joy': ['celebration', 'friendship', 'achievement', 'new beginnings', 'community'],
            'sadness': ['healing', 'memory', 'loss', 'redemption', 'family bonds'],
            'anger': ['justice', 'transformation', 'standing up', 'social change', 'empowerment'],
            'fear': ['survival', 'overcoming', 'facing the unknown', 'courage', 'protection'],
            'surprise': ['discovery', 'revelation', 'adventure', 'unexpected turns', 'mystery'],
            'love': ['connection', 'sacrifice', 'devotion', 'relationships', 'commitment'],
            'excitement': ['adventure', 'exploration', 'achievement', 'competition', 'dreams']
        }
        
        # Narrative tones
        narrative_tones = {
            'joy': 'uplifting and optimistic' if intensity > 5 else 'gentle and heartwarming',
            'sadness': 'poignant and reflective' if intensity > 5 else 'melancholic and subtle',
            'anger': 'intense and confrontational' if intensity > 6 else 'determined and focused',
            'fear': 'suspenseful and tense' if intensity > 6 else 'mysterious and atmospheric',
            'surprise': 'dynamic and unexpected' if intensity > 5 else 'curious and intriguing',
            'love': 'passionate and romantic' if intensity > 6 else 'tender and intimate',
            'excitement': 'energetic and fast-paced' if intensity > 6 else 'enthusiastic and engaging'
        }
        
        return {
            'description': emotion_descriptions.get(emotion, f"You're experiencing {emotion} with intensity {intensity}/10."),
            'themes': emotion_themes.get(emotion, ['personal growth', 'human nature', 'life lessons']),
            'tone': narrative_tones.get(emotion, 'balanced and engaging')
        }
    
    def get_emotion_distribution(self, emotions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get the distribution of emotions"""
        if not emotions:
            return {}
        
        distribution = {}
        total_score = sum(emotion['score'] for emotion in emotions)
        
        for emotion in emotions:
            distribution[emotion['label']] = (emotion['score'] / total_score) * 100
        
        return distribution
    
    def detect_complex_emotions(self, input_text: str) -> Dict[str, Any]:
        """Detect complex emotional states like mixed emotions"""
        emotions = self._analyze_text_emotion(input_text)
        
        if len(emotions) < 2:
            return {'is_complex': False, 'primary_emotion': emotions[0]['label'] if emotions else 'neutral'}
        
        # Check if top emotions have similar scores (indicating mixed emotions)
        top_emotion_score = emotions[0]['score']
        second_emotion_score = emotions[1]['score']
        
        is_complex = (top_emotion_score - second_emotion_score) < 0.2
        
        if is_complex:
            return {
                'is_complex': True,
                'primary_emotion': emotions[0]['label'],
                'secondary_emotion': emotions[1]['label'],
                'emotion_mix': f"{emotions[0]['label']} with {emotions[1]['label']}",
                'complexity_score': 1 - (top_emotion_score - second_emotion_score)
            }
        
        return {
            'is_complex': False,
            'primary_emotion': emotions[0]['label']
        }