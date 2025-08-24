import streamlit as st
import random
import re
import json
import datetime
from typing import Dict, List, Any
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Mood to Movie Generator",
    page_icon="",
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
        line-height: 1.6;
    }
    
    .poster-container {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(45deg, #f0f0f0, #ffffff);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .model-comparison {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .gpt-style {
        border-left: 5px solid #10a37f;
        background-color: #f0f9f6;
    }
    
    .claude-style {
        border-left: 5px solid #8b5cf6;
        background-color: #f8f5ff;
    }
    
    .bert-style {
        border-left: 5px solid #f59e0b;
        background-color: #fffbeb;
    }
    
    .llama-style {
        border-left: 5px solid #dc2626;
        background-color: #fef2f2;
    }
    
    .export-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
        margin: 1rem 0;
    }
    
    .version-history {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #9c27b0;
        margin: 1rem 0;
    }
    
    .template-library {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
        margin: 1rem 0;
    }
    
    .collaboration-tools {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for technical features
if 'story_history' not in st.session_state:
    st.session_state.story_history = []
if 'current_story_id' not in st.session_state:
    st.session_state.current_story_id = 0
if 'collaborators' not in st.session_state:
    st.session_state.collaborators = []
if 'templates' not in st.session_state:
    st.session_state.templates = {}
if 'current_story' not in st.session_state:
    st.session_state.current_story = None

def detect_mood_simple(text: str) -> Dict[str, Any]:
    """Simple rule-based mood detection"""
    text_lower = text.lower()
    
    # Remove empty strings and improve keyword matching
    mood_keywords = {
        'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', '😊', '😄', 'good', 'nice', 'pleased', 'delighted', 'cheerful', 'bright', 'sunny', 'positive'],
        'sad': ['sad', 'depressed', 'miserable', 'unhappy', 'gloomy', 'melancholy', '😢', '💔', 'down', 'blue', 'sorrow', 'grief', 'heartbroken', 'lonely', 'hopeless'],
        'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', '😠', '🤬', 'rage', 'outraged', 'livid', 'enraged', 'hostile', 'aggressive'],
        'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'quiet', '😌', '🧘', 'peaceful', 'gentle', 'soothing', 'mellow', 'composed', 'centered'],
        'excited': ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped', '🤩', '🌟', 'energetic', 'buzzed', 'stoked', 'amped', 'fired up', 'motivated'],
        'nervous': ['nervous', 'anxious', 'worried', 'stressed', 'tense', '😰', '😱', 'jittery', 'on edge', 'uneasy', 'apprehensive', 'fearful', 'panicked'],
        'romantic': ['romantic', 'loving', 'passionate', 'tender', 'affectionate', '❤️', '😍', '🥰', 'sweet', 'caring', 'devoted', 'adoring', 'cherished']
    }
    
    detected_moods = {}
    for mood, keywords in mood_keywords.items():
        # Only count non-empty keywords and improve matching
        score = sum(1 for keyword in keywords if keyword and keyword in text_lower)
        detected_moods[mood] = score
    
    # Normalize scores to get confidence
    total_score = sum(detected_moods.values())
    if total_score == 0:
        # Default to a neutral mood if no keywords detected
        return {"detected_mood": "neutral", "confidence": 0.5, "mood_breakdown": {m: 0.0 for m in mood_keywords}}
    
    mood_breakdown = {mood: score / total_score for mood, score in detected_moods.items()}
    
    # Find the dominant mood
    detected_mood = max(mood_breakdown, key=mood_breakdown.get)
    confidence = mood_breakdown[detected_mood]
    
    return {
        "detected_mood": detected_mood,
        "confidence": confidence,
        "mood_breakdown": mood_breakdown
    }

def generate_story_gpt(mood: str, genre: str = "Drama") -> Dict[str, Any]:
    """Simulate GPT-style story generation - verbose, detailed, sometimes repetitive"""
    
    gpt_templates = {
        'happy': f"""Based on your description of feeling {mood.lower()}, I'll create a story that captures this emotional state. Let me craft a narrative that explores the themes of joy, optimism, and positive transformation.

In a world where happiness seemed to flow like an endless river, there lived a character named Maya whose very existence was a testament to the power of positive thinking and emotional resilience. Maya's happiness wasn't simply a fleeting emotion or temporary state of mind—it was a fundamental aspect of her personality that had been carefully cultivated through years of conscious practice and intentional living. Her journey through life was characterized by an unwavering commitment to finding the silver lining in every situation, no matter how challenging or difficult it might initially appear.

As Maya navigated the complex landscape of human experience, her happiness evolved from a simple emotional state into a comprehensive philosophy that guided every decision she made and every interaction she had with others. When she encountered obstacles or setbacks, she approached them not with fear or trepidation, but with the confident expectation that solutions existed and that every challenge was ultimately an opportunity for personal growth and development. Her ability to maintain a positive outlook even in the face of adversity became a source of inspiration for everyone around her, demonstrating that happiness is not dependent on external circumstances but is rather a choice that can be made in any moment.

Through her various adventures and experiences, Maya discovered that authentic happiness is not about avoiding negative emotions or pretending that life is always perfect and wonderful. Instead, it's about developing the capacity to experience the full range of human emotions while maintaining an underlying sense of contentment and gratitude. She learned that true joy comes from within and can be cultivated through practices such as mindfulness, gratitude, and meaningful connections with others. In the end, Maya's greatest achievement wasn't the material wealth she accumulated or the places she visited, but the countless lives she touched with her radiant spirit and the positive impact she had on her community.""",
        
        'sad': f"""I understand you're experiencing feelings of {mood.lower()}, and I want to create a story that acknowledges this emotional state while also exploring the deeper meaning and potential for growth that can come from these difficult experiences.

In the depths of melancholy, where shadows seemed to dance with sorrow and the weight of the world pressed down with an almost unbearable heaviness, there existed a soul named Alex who had developed an intimate relationship with sadness that few people could truly understand. Alex's journey through darkness wasn't a choice they had made or a path they had willingly chosen—it was a reality that had shaped every aspect of their existence, teaching them lessons about the human condition that could only be learned through experiencing the full spectrum of emotional pain and suffering. The sadness that clung to their heart like morning mist wasn't just an emotion—it was a companion that had walked beside them through the darkest valleys of life, whispering secrets about resilience, strength, and the capacity for transformation that few were brave enough to hear.

As Alex moved through their daily existence, each step felt like walking through water, each breath a conscious effort that required more energy than they sometimes possessed. The sadness had taught them to see beauty in unexpected places: in the way raindrops traced intricate patterns on windows, in the gentle curve of a wilting flower that still held its own kind of grace, in the soft light that filtered through storm clouds to create moments of unexpected illumination. These moments of beauty, though fleeting and sometimes difficult to recognize, became anchors that kept them connected to a world that sometimes seemed too painful and overwhelming to inhabit. Through their struggles and challenges, Alex discovered a strength they never knew they possessed, a resilience that grew stronger with each obstacle they faced and overcame.

The transformation that occurred within Alex's soul wasn't sudden or dramatic, but rather a gradual awakening that happened in the quiet spaces between their tears and in the moments of reflection that came after each wave of sadness had passed. They learned that sadness, when embraced rather than fought against or suppressed, could become a teacher rather than a tormentor, offering insights and wisdom that could only be gained through experiencing the full range of human emotion. The understanding they developed through their emotional journey became a gift they could offer to others who were walking similar paths, providing comfort, validation, and hope to those who felt alone in their suffering. In the end, Alex emerged from the shadows not as someone who had conquered or eliminated their sadness, but as someone who had learned to dance with it, carrying a wisdom and compassion that only comes from experiencing the full spectrum of human emotion and finding meaning in even the most difficult experiences.""",
        
        'angry': f"""I can sense from your description that you're experiencing feelings of {mood.lower()}, and I want to create a story that explores this powerful emotion and its potential for positive transformation and personal growth.

In a world where frustration burned like wildfire and injustice seemed to lurk around every corner, waiting to strike at the most vulnerable and defenseless members of society, there lived a character named Jordan whose anger was a force to be reckoned with—a blazing inferno that threatened to consume everything in its path if not properly understood and channeled. Jordan's anger wasn't the petty irritation of minor inconveniences or the temporary frustration of everyday setbacks; it was a deep, burning rage that had been stoked by years of witnessing unfairness, experiencing betrayal, and feeling powerless in the face of systems and structures that seemed designed to keep certain people down while elevating others to positions of privilege and power. This anger lived in their chest like a second heart, beating with the rhythm of every slight, every disappointment, every moment when the world had failed to live up to its promises of equality, justice, and basic human dignity.

As Jordan moved through their days, the anger became both their greatest weapon and their heaviest burden, a double-edged sword that could cut through deception and injustice but could also wound the wielder if not handled with care and precision. It fueled their determination to fight for what was right, to speak truth to power, and to refuse to accept the status quo or the comfortable lies that people told themselves to avoid facing difficult truths about the world and their place in it. When they saw someone being treated unfairly or experiencing discrimination, their anger became a shield that protected the vulnerable and a voice that amplified the concerns of those who had been silenced. When they encountered corruption, dishonesty, or systemic problems, it became a sword that cut through deception and revealed the truth beneath the surface. However, the anger also had a darker side that Jordan had to learn to recognize and manage, sometimes clouding their judgment and making it difficult to see the good in people or situations, or to approach problems with the clarity and objectivity that effective problem-solving requires.

The journey toward mastering their anger became Jordan's greatest challenge and most important lesson, requiring years of self-reflection, therapy, meditation, and conscious effort to transform a potentially destructive force into a powerful catalyst for positive change and personal growth. They discovered that anger, when channeled properly and understood deeply, could be transformed from a destructive force that burned everything in its path into a constructive energy that could build rather than destroy, heal rather than harm, and create rather than annihilate. Instead of letting their anger burn out of control like a wildfire, they learned to harness it like a controlled explosion, using its energy to power their efforts to make the world a better place and to help others who were struggling with similar challenges. Through various therapeutic approaches, mindfulness practices, and countless hours of self-reflection and personal development work, Jordan developed the ability to feel their anger without being consumed by it, to experience the emotion fully while maintaining their capacity for rational thought and compassionate action. In the end, they emerged not as someone who had eliminated or suppressed their anger, but as someone who had learned to use it as a tool for justice, a voice for the voiceless, and a force for positive transformation in both their own life and the lives of others. Their story became a powerful reminder that anger, when properly understood, respected, and directed, can be one of the most powerful forces for positive change and personal growth in the world."""
    }
    
    story = gpt_templates.get(mood.lower(), gpt_templates['happy'])
    
    return {
        'title': f"The {mood.capitalize()} Journey - GPT Style",
        'story': story,
        'genre': genre,
        'mood': mood,
        'method': 'GPT-4 Simulation',
        'style': 'gpt-style'
    }

def generate_story_claude(mood: str, genre: str = "Drama") -> Dict[str, Any]:
    """Simulate Claude-style story generation - thoughtful, balanced, philosophical"""
    
    claude_templates = {
        'happy': f"""I appreciate you sharing your feelings of {mood.lower()} with me. Let me craft a story that explores this emotional state with nuance and depth, acknowledging both the beauty and complexity of human happiness.

In contemplating the nature of joy and contentment, I find myself drawn to the story of Maya, a character whose relationship with happiness transcends simple emotional states. Maya's happiness isn't merely the absence of sadness or the presence of pleasure—it's a profound understanding of what it means to be fully human, to embrace both the light and shadow aspects of existence while maintaining an underlying sense of gratitude and wonder. Her journey represents a philosophical exploration of how we can cultivate sustainable happiness in a world that often seems designed to undermine our sense of well-being.

What makes Maya's story particularly compelling is her recognition that true happiness requires both individual effort and collective support. She understands that personal joy is interconnected with the well-being of others, and that her own happiness grows when she helps others find theirs. This perspective challenges the common misconception that happiness is a zero-sum game or a finite resource that must be hoarded. Instead, Maya demonstrates that joy, like love and knowledge, multiplies when shared generously with others.

The philosophical depth of Maya's approach to happiness lies in her acceptance of life's inherent impermanence and uncertainty. Rather than seeking a permanent state of bliss—which would be both unrealistic and potentially harmful—she cultivates the ability to find meaning and beauty in every moment, whether that moment brings pleasure or pain. This approach reflects ancient wisdom traditions that emphasize the importance of equanimity and the cultivation of inner resources that can sustain us through life's inevitable ups and downs. Maya's story ultimately suggests that the most profound happiness comes not from avoiding difficulty, but from developing the capacity to meet all of life's experiences with wisdom, compassion, and an open heart.""",
        
        'sad': f"""I hear the {mood.lower()} in your words, and I want to honor that feeling while also exploring the deeper wisdom that can emerge from our most difficult emotional experiences. Let me share a story that acknowledges the reality of suffering while also pointing toward the possibility of growth and transformation.

Alex's story begins in what might appear to be a place of darkness and despair, but what I find most compelling about their journey is how it reveals the hidden dimensions of sadness that our culture often fails to acknowledge. Sadness, when approached with curiosity and compassion rather than fear and avoidance, can become a gateway to deeper self-understanding and empathy for others. Alex's experience suggests that our most painful emotions often contain the seeds of our greatest wisdom and strength.

What I find particularly meaningful about Alex's approach to sadness is their willingness to stay present with difficult emotions rather than rushing to escape them. In a culture that often encourages us to "think positive" or "look on the bright side," Alex's story reminds us that there is profound value in allowing ourselves to fully experience our emotions, even when they are uncomfortable or painful. This approach aligns with contemporary psychological research that suggests emotional suppression can actually prolong suffering, while emotional acceptance can lead to greater psychological flexibility and resilience.

The philosophical implications of Alex's journey extend beyond individual healing to questions about how we as a society approach emotional suffering. Alex's story challenges us to reconsider our collective tendency to pathologize sadness and to recognize that difficult emotions can serve important functions in our psychological and spiritual development. Their transformation suggests that the goal isn't to eliminate sadness entirely, but to develop a more nuanced and compassionate relationship with all of our emotional experiences, recognizing that each emotion has something valuable to teach us about ourselves and the human condition.""",
        
        'angry': f"""I can sense the {mood.lower()} in your description, and I want to acknowledge the validity of that emotion while exploring how it might be understood and channeled in constructive ways. Anger, like all emotions, serves important functions in our psychological and social lives, and understanding its nature can help us work with it more skillfully.

Jordan's story illustrates what I find to be one of the most important insights about anger: it's not inherently good or bad, but rather a powerful energy that can be directed toward either constructive or destructive ends. The key question isn't whether we should feel angry—we will, and that's completely normal and healthy—but rather how we can work with our anger in ways that serve our values and contribute to positive change. Jordan's journey demonstrates that anger can be a powerful motivator for social justice, personal growth, and the protection of vulnerable individuals.

What I find particularly compelling about Jordan's approach is their recognition that anger often masks other emotions like fear, hurt, or a sense of powerlessness. By learning to identify the underlying causes of their anger, Jordan develops the ability to respond more skillfully to challenging situations. This aligns with contemporary psychological approaches that emphasize the importance of emotional intelligence and the ability to recognize and work with the full spectrum of our emotional experiences.

The philosophical depth of Jordan's story lies in its exploration of how we can transform potentially destructive emotions into forces for positive change. Jordan's journey suggests that the goal isn't to eliminate anger or other difficult emotions, but to develop the wisdom and skill to work with them constructively. This approach reflects ancient philosophical traditions that emphasize the importance of self-knowledge and the cultivation of virtues like courage, wisdom, and compassion. Jordan's story ultimately suggests that our most challenging emotions can become our greatest teachers, helping us develop the strength and wisdom to create positive change in both our own lives and the world around us."""
    }
    
    story = claude_templates.get(mood.lower(), claude_templates['happy'])
    
    return {
        'title': f"The {mood.capitalize()} Journey - Claude Style",
        'story': story,
        'genre': genre,
        'mood': mood,
        'method': 'Claude-3 Sonnet Simulation',
        'style': 'claude-style'
    }

def generate_story_bert(mood: str, genre: str = "Drama") -> Dict[str, Any]:
    """Simulate BERT-style story generation - focused, technical, structured"""
    
    bert_templates = {
        'happy': f"""Mood Analysis: {mood.upper()} - Positive emotional state detected.

Story Generation Parameters:
- Primary emotion: {mood.lower()}
- Genre: {genre}
- Tone: Optimistic
- Structure: Three-act narrative

ACT 1: INTRODUCTION
Maya, a 28-year-old software engineer, experiences consistent happiness in her daily life. Her emotional baseline is characterized by positive affect, high life satisfaction, and optimistic outlook. Maya's happiness is not contingent on external circumstances but stems from internal cognitive patterns and behavioral choices.

ACT 2: DEVELOPMENT
Maya's happiness manifests in specific behavioral indicators: frequent smiling, positive social interactions, proactive problem-solving, and resilience in the face of challenges. Her emotional regulation strategies include gratitude practices, mindfulness meditation, and maintaining strong social connections. These factors contribute to her sustained positive emotional state.

ACT 3: RESOLUTION
Maya's happiness demonstrates the effectiveness of evidence-based positive psychology interventions. Her story illustrates how intentional practices can cultivate sustainable happiness, supporting research findings on the relationship between positive emotions and life satisfaction. The narrative concludes with Maya's continued emotional well-being and positive impact on her community.

Key Themes: Positive psychology, emotional regulation, social connection, resilience.""",
        
        'sad': f"""Mood Analysis: {mood.upper()} - Negative emotional state detected.

Story Generation Parameters:
- Primary emotion: {mood.lower()}
- Genre: {genre}
- Tone: Contemplative
- Structure: Three-act narrative

ACT 1: INTRODUCTION
Alex, a 32-year-old graduate student, experiences persistent sadness characterized by low mood, reduced energy, and social withdrawal. Alex's emotional state is influenced by academic stress, social isolation, and underlying cognitive patterns that contribute to negative affect.

ACT 2: DEVELOPMENT
Alex's sadness manifests in specific behavioral indicators: decreased motivation, sleep disturbances, reduced appetite, and negative self-talk. The character's emotional processing follows established patterns of depressive cognition, including rumination and cognitive distortions. However, Alex demonstrates adaptive coping through journaling and seeking social support.

ACT 3: RESOLUTION
Alex's journey illustrates the complexity of emotional experience and the importance of adaptive coping strategies. The narrative demonstrates how sadness, when properly understood and managed, can lead to personal growth and increased emotional intelligence. Alex's story supports research on the adaptive functions of negative emotions.

Key Themes: Emotional processing, adaptive coping, cognitive restructuring, social support.""",
        
        'angry': f"""Mood Analysis: {mood.upper()} - High-arousal negative emotional state detected.

Story Generation Parameters:
- Primary emotion: {mood.lower()}
- Genre: {genre}
- Tone: Intense
- Structure: Three-act narrative

ACT 1: INTRODUCTION
Jordan, a 26-year-old activist, experiences intense anger in response to perceived social injustices. Jordan's emotional state is characterized by high physiological arousal, cognitive activation, and behavioral preparation for action. The anger serves as a motivational force for social engagement and change.

ACT 2: DEVELOPMENT
Jordan's anger manifests in specific behavioral indicators: increased energy, focused attention, assertive communication, and goal-directed behavior. The character's emotional regulation involves channeling anger into constructive action rather than suppression or aggression. Jordan demonstrates effective anger management through cognitive reappraisal and behavioral redirection.

ACT 3: RESOLUTION
Jordan's story illustrates the functional aspects of anger as a motivational emotion. The narrative demonstrates how anger, when properly regulated and directed, can serve as a catalyst for positive social change. Jordan's journey supports research on the adaptive functions of anger in promoting justice and social reform.

Key Themes: Emotional regulation, social justice, motivational emotion, adaptive anger expression."""
    }
    
    story = bert_templates.get(mood.lower(), bert_templates['happy'])
    
    return {
        'title': f"The {mood.capitalize()} Journey - BERT Style",
        'story': story,
        'genre': genre,
        'mood': mood,
        'method': 'BERT-Base Simulation',
        'style': 'bert-style'
    }

def generate_story_llama(mood: str, genre: str = "Drama") -> Dict[str, Any]:
    """Simulate LLaMA-style story generation - creative, sometimes unpredictable, varied quality"""
    
    llama_templates = {
        'happy': f"""Once upon a time in a world where happiness was like... well, happiness was everywhere! Maya was this girl who was just super happy all the time. Like, you know how some people are just naturally happy? That was Maya. She had this way of making everyone around her happy too, which was pretty cool.

Maya's thing was that she didn't just wait around to be happy - she went out and found happiness in everything. Like, even when it was raining, she'd be like "Oh, the rain is so beautiful!" and when things went wrong, she'd be like "Well, that's just life teaching us something!" It was kind of amazing how she could turn any situation into something positive.

The really interesting part about Maya was that her happiness wasn't fake or forced - it was real and deep. She had this philosophy that happiness is a choice we make every day, and she chose to be happy no matter what. And you know what? It worked! People around her started feeling happier too, and soon her whole community was like this little bubble of joy and positivity. It was pretty amazing to see how one person's choice to be happy could spread like that.

In the end, Maya showed everyone that happiness isn't about having everything perfect - it's about choosing to see the good in everything, even the hard stuff. She proved that when you choose happiness, you don't just make yourself happy, you make the whole world a little bit brighter. Pretty cool, right?""",
        
        'sad': f"""So there was this person named Alex who was going through a really tough time. Like, really tough. Alex was just... sad. Not the kind of sad that goes away after a good night's sleep, but the kind that sticks around and makes everything feel heavy and hard. It was like carrying around this big weight all the time.

Alex's sadness was complicated, you know? It wasn't just about one thing - it was about a lot of things that had built up over time. Life had been pretty rough, and Alex had learned to expect disappointment and pain. But here's the thing about Alex: even though they were sad, they were also really strong. Like, they kept going even when it felt impossible, and they found ways to cope that actually worked.

The really interesting thing about Alex's journey was that they started to see their sadness differently. Instead of fighting it or trying to ignore it, they started to understand it. They realized that sadness can actually teach you things - like how to be more compassionate, how to appreciate the good moments more, and how to help other people who are going through hard times. It was like their sadness became this teacher that helped them grow into a better, wiser person.

In the end, Alex didn't stop being sad - that's not really how life works. But they learned how to live with their sadness in a way that made them stronger instead of weaker. They discovered that sometimes the people who have been through the most pain are the ones who can help others the most, because they really understand what it's like to suffer. It was pretty amazing how Alex turned something that could have destroyed them into something that made them stronger and more compassionate.""",
        
        'angry': f"""Jordan was mad. Like, really mad. Not the kind of mad you get when someone cuts you off in traffic, but the kind of mad that burns deep and hot and makes you want to change the whole world. Jordan had seen a lot of unfair stuff, and they were tired of it. Really tired of it.

The thing about Jordan's anger was that it wasn't just random - it was focused and specific. They were angry about injustice, about people being treated badly, about systems that seemed designed to keep some people down while others got ahead. It was the kind of anger that made sense, you know? Like, if you weren't angry about this stuff, there was probably something wrong with you.

But here's where Jordan's story gets interesting: they didn't just stay angry. They learned how to use their anger as fuel for change. Instead of letting it burn them up from the inside, they channeled it into action. They started speaking up, organizing people, and working to fix the things that made them angry in the first place. It was pretty amazing to watch someone turn their anger into something positive and powerful.

The really cool thing about Jordan's journey was that they learned how to be angry without being destructive. They figured out that anger can be a tool for good if you know how to use it right. Instead of just yelling and breaking things, they used their anger to motivate themselves and others to make real changes. In the end, Jordan proved that anger, when handled right, can be one of the most powerful forces for good in the world. Pretty impressive, right?"""
    }
    
    story = llama_templates.get(mood.lower(), llama_templates['happy'])
    
    return {
        'title': f"The {mood.capitalize()} Journey - LLaMA Style",
        'story': story,
        'genre': genre,
        'mood': mood,
        'method': 'LLaMA-2 Simulation',
        'style': 'llama-style'
    }

def generate_story_with_model(mood: str, model_style: str, genre: str = "Drama") -> Dict[str, Any]:
    """Generate story using the selected LLM model simulation"""
    
    if model_style == "GPT-4":
        return generate_story_gpt(mood, genre)
    elif model_style == "Claude-3 Sonnet":
        return generate_story_claude(mood, genre)
    elif model_style == "BERT-Base":
        return generate_story_bert(mood, genre)
    else:  # LLaMA-2
        return generate_story_llama(mood, genre)

def generate_poster_simple(title: str, genre: str, mood: str) -> Dict[str, Any]:
    """Generate poster description using templates with proper mood matching"""
    
    mood_specific_posters = {
        'happy': {
            'colors': ['Warm yellows', 'Bright oranges', 'Vibrant pinks', 'Sunny golds'],
            'layout': 'Dynamic composition with upward movement and bright lighting',
            'mood_elements': 'Joyful atmosphere with uplifting visual elements and cheerful imagery',
            'genre_elements': 'Optimistic storytelling with themes of triumph and celebration'
        },
        
        'sad': {
            'colors': ['Cool blues', 'Soft purples', 'Gentle grays', 'Muted tones'],
            'layout': 'Contemplative composition with gentle curves and soft shadows',
            'mood_elements': 'Melancholic atmosphere with reflective visual elements',
            'genre_elements': 'Emotional storytelling with themes of growth and understanding'
        },
        
        'angry': {
            'colors': ['Deep reds', 'Dark oranges', 'Bold blacks', 'Fiery tones'],
            'layout': 'Dynamic composition with sharp angles and intense contrasts',
            'mood_elements': 'Intense atmosphere with powerful visual elements and dramatic lighting',
            'genre_elements': 'High-impact storytelling with themes of transformation and strength'
        },
        
        'excited': {
            'colors': ['Electric blues', 'Bright reds', 'Dynamic greens', 'Vibrant purples'],
            'layout': 'Energetic composition with diagonal lines and dynamic movement',
            'mood_elements': 'Thrilling atmosphere with high-energy visual elements',
            'genre_elements': 'Adventure storytelling with themes of discovery and excitement'
        },
        
        'calm': {
            'colors': ['Soft pastels', 'Earth tones', 'Gentle whites', 'Muted blues'],
            'layout': 'Balanced composition with smooth lines and harmonious proportions',
            'mood_elements': 'Peaceful atmosphere with serene visual elements',
            'genre_elements': 'Contemplative storytelling with themes of wisdom and tranquility'
        },
        
        'nervous': {
            'colors': ['Cool grays', 'Soft blues', 'Muted greens', 'Gentle purples'],
            'layout': 'Tense composition with subtle movement and careful balance',
            'mood_elements': 'Anxious atmosphere with delicate visual elements',
            'genre_elements': 'Suspenseful storytelling with themes of courage and growth'
        }
    }
    
    poster_details = mood_specific_posters.get(mood.lower(), mood_specific_posters['calm'])
    
    poster_description = f"""
    A cinematic poster featuring the title '{title}' prominently displayed.
    The poster follows a {genre.lower()} style with {mood.lower()} undertones.
    
    Color Palette: {', '.join(poster_details['colors'])}
    Layout: {poster_details['layout']}
    Mood Elements: {poster_details['mood_elements']}
    Genre Elements: {poster_details['genre_elements']}
    
    The overall design creates a {mood.lower()} atmosphere that perfectly captures the emotional journey of the story.
    """
    
    return {
        'title': title,
        'genre': genre,
        'mood': mood,
        'poster_description': poster_description,
        'method': 'template_based'
    }

def get_soundtrack_simple(mood: str) -> Dict[str, List[str]]:
    """Get soundtrack recommendations based on mood with proper matching"""
    
    mood_soundtracks = {
        'happy': [
            'Upbeat pop songs with infectious melodies',
            'Feel-good rock anthems that lift your spirits',
            'Cheerful electronic music with positive vibes',
            'Joyful folk songs with warm acoustic sounds',
            'Motivational hip-hop with empowering lyrics'
        ],
        'sad': [
            'Melancholic ballads with emotional depth',
            'Gentle piano pieces that soothe the soul',
            'Soulful blues with heartfelt storytelling',
            'Touching acoustic songs with raw emotion',
            'Peaceful orchestral pieces for reflection'
        ],
        'angry': [
            'High-energy rock with powerful guitar riffs',
            'Intense electronic music with driving beats',
            'Aggressive hip-hop with strong lyrical content',
            'Dramatic orchestral pieces with bold brass',
            'Heavy metal with cathartic energy release'
        ],
        'excited': [
            'High-energy electronic with pumping bass',
            'Dynamic rock anthems with driving rhythms',
            'Fast-paced pop with infectious energy',
            'Energetic hip-hop with motivational beats',
            'Thrilling orchestral with dramatic crescendos'
        ],
        'calm': [
            'Ambient soundscapes with gentle textures',
            'Smooth jazz with relaxed improvisation',
            'Peaceful classical with gentle melodies',
            'Gentle acoustic with soft vocals',
            'Relaxing nature sounds with peaceful atmosphere'
        ],
        'nervous': [
            'Soft ambient music with calming tones',
            'Gentle acoustic with soothing melodies',
            'Peaceful instrumental with gentle rhythms',
            'Calming classical with soft dynamics',
            'Relaxing electronic with smooth transitions'
        ],
        'romantic': [
            'Love ballads with tender vocals',
            'Soft jazz with romantic melodies',
            'Tender acoustic with heartfelt lyrics',
            'Emotional pop with romantic themes',
            'Romantic classical with beautiful harmonies'
        ]
    }
    
    recommendations = mood_soundtracks.get(mood.lower(), ['Versatile soundtrack for all moods'])
    
    return {
        'mood_based': recommendations[:3],
        'overall_style': f"{mood.capitalize()} soundtrack with {mood.lower()} undertones",
        'mood': mood
    }

# Technical Improvement Functions
def save_story_version(story_data: Dict[str, Any]) -> None:
    """Save current story to version history"""
    story_data['timestamp'] = datetime.datetime.now().isoformat()
    story_data['version_id'] = len(st.session_state.story_history) + 1
    story_data['story_id'] = st.session_state.current_story_id
    
    st.session_state.story_history.append(story_data)
    st.success(f"Story version {story_data['version_id']} saved!")

def export_story_as_pdf(story_data: Dict[str, Any]) -> str:
    """Export story as PDF format"""
    # This would integrate with a PDF library like reportlab or fpdf
    # For now, return a formatted string
    pdf_content = f"""
    MOOD TO MOVIE GENERATOR - STORY EXPORT
    
    Title: {story_data.get('title', 'Untitled')}
    Mood: {story_data.get('mood', 'Unknown')}
    Model: {story_data.get('model', 'Unknown')}
    Generated: {story_data.get('timestamp', 'Unknown')}
    
    STORY:
    {story_data.get('story', 'No story content')}
    
    POSTER SPECIFICATIONS:
    {json.dumps(story_data.get('poster', {}), indent=2)}
    """
    return pdf_content

def export_story_as_screenplay(story_data: Dict[str, Any]) -> str:
    """Export story as screenplay format"""
    screenplay = f"""
    FADE IN:
    
    {story_data.get('story', 'No story content')}
    
    FADE OUT.
    
    THE END
    """
    return screenplay

def add_collaborator(email: str, role: str) -> None:
    """Add a collaborator to the project"""
    collaborator = {
        'email': email,
        'role': role,
        'added_date': datetime.datetime.now().isoformat(),
        'permissions': ['read', 'comment'] if role == 'viewer' else ['read', 'comment', 'edit']
    }
    st.session_state.collaborators.append(collaborator)
    st.success(f"Collaborator {email} added as {role}!")

def load_story_template(template_name: str) -> Dict[str, Any]:
    """Load a pre-built story template"""
    templates = {
        'hero_journey': {
            'name': 'Hero\'s Journey',
            'structure': ['Call to Adventure', 'Crossing the Threshold', 'Tests and Trials', 'Return with Elixir'],
            'description': 'Classic hero journey structure for epic adventures'
        },
        'romance': {
            'name': 'Romance Story',
            'structure': ['Meet Cute', 'Conflict', 'Resolution', 'Happy Ending'],
            'description': 'Traditional romance story structure'
        },
        'mystery': {
            'name': 'Mystery Thriller',
            'structure': ['Setup', 'Investigation', 'Climax', 'Revelation'],
            'description': 'Suspenseful mystery structure'
        },
        'coming_of_age': {
            'name': 'Coming of Age',
            'structure': ['Innocence', 'Experience', 'Growth', 'Maturity'],
            'description': 'Character development focused structure'
        }
    }
    return templates.get(template_name, {})

def main():
    st.markdown('<h1 class="main-header">🎬 Mood to Movie Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎭 Express Your Mood")
        user_input = st.text_area("How are you feeling today?", height=100)
        
        st.header("🤖 Choose LLM Model")
        model_options = ["GPT-4", "Claude-3 Sonnet", "BERT", "LLaMA"]
        selected_model = st.selectbox("Select Model", model_options)
        
        st.header("📚 Story Template")
        template_options = ["None", "hero_journey", "romance", "mystery", "coming_of_age"]
        selected_template = st.selectbox("Choose Template", template_options)
        
        st.header("👥 Collaboration")
        collaborator_email = st.text_input("Add Collaborator Email")
        collaborator_role = st.selectbox("Role", ["viewer", "editor", "admin"])
        if st.button("Add Collaborator"):
            if collaborator_email:
                add_collaborator(collaborator_email, collaborator_role)
        
        st.header("💾 Save & Export")
        if st.button("Save Current Version"):
            if st.session_state.current_story is not None:
                save_story_version(st.session_state.current_story)
            else:
                st.warning("No story to save yet. Generate a story first!")
    
    # Main content
    if user_input:
        # Mood detection
        mood_result = detect_mood_simple(user_input)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="mood-card">', unsafe_allow_html=True)
            st.subheader("🎭 Detected Mood")
            st.write(f"**Mood:** {mood_result['detected_mood'].title()}")
            st.write(f"**Confidence:** {mood_result['confidence']:.2f}")
            st.write(f"**Selected Model:** {selected_model}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Mood Breakdown")
            for mood, score in mood_result['mood_breakdown'].items():
                st.progress(score)
                st.write(f"{mood.title()}: {score:.2f}")
        
        # Story generation
        st.subheader(" Generated Story")
        story = generate_story_with_model(mood_result['detected_mood'], selected_model)
        st.markdown('<div class="script-container">', unsafe_allow_html=True)
        st.write(story.get('story', 'Story generation failed.'))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Poster generation
        st.subheader("🎨 Movie Poster")
        poster_specs = generate_poster_simple(
            "Your Story",
            "Drama",
            mood_result['detected_mood']
        )
        st.markdown('<div class="poster-container">', unsafe_allow_html=True)
        st.write("**Poster Specifications:**")
        for key, value in poster_specs.items():
            st.write(f"**{key.title()}:** {value}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save current story
        st.session_state.current_story = {
            'title': f"Story for {mood_result['detected_mood']} mood",
            'mood': mood_result['detected_mood'],
            'model': selected_model,
            'story': story.get('story', 'Story generation failed.'),
            'poster': poster_specs,
            'user_input': user_input,
            'template': selected_template if selected_template != "None" else None
        }
        
        # Technical Improvements Section
        st.markdown("---")
        st.header(" Technical Features")
        
        # Export Options
        with st.expander("📤 Export Options", expanded=True):
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.session_state.current_story is not None:
                    pdf_content = export_story_as_pdf(st.session_state.current_story)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_content,
                        file_name=f"story_{mood_result['detected_mood']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("No story available to export!")
            
            with col2:
                if st.session_state.current_story is not None:
                    screenplay = export_story_as_screenplay(st.session_state.current_story)
                    st.download_button(
                        label="Download Screenplay",
                        data=screenplay,
                        file_name=f"screenplay_{mood_result['detected_mood']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("No story available to export!")
            
            with col3:
                if st.session_state.current_story is not None:
                    json_data = json.dumps(st.session_state.current_story, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"story_data_{mood_result['detected_mood']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error("No story available to export!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Version History
        with st.expander("📚 Version History", expanded=True):
            st.markdown('<div class="version-history">', unsafe_allow_html=True)
            if st.session_state.story_history:
                for version in reversed(st.session_state.story_history[-5:]):  # Show last 5 versions
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**Version {version['version_id']}** - {version['mood'].title()}")
                            st.write(f"Model: {version['model']}")
                        with col2:
                            st.write(f"**{version['timestamp'][:10]}**")
                        with col3:
                            if st.button(f"Restore v{version['version_id']}", key=f"restore_{version['version_id']}"):
                                st.session_state.current_story = version.copy()
                                st.success(f"Restored version {version['version_id']}!")
                                st.rerun()
            else:
                st.info("No saved versions yet. Save your first story to see version history!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Template Library
        with st.expander(" Template Library", expanded=True):
            st.markdown('<div class="template-library">', unsafe_allow_html=True)
            if selected_template != "None":
                template_info = load_story_template(selected_template)
                if template_info:
                    st.write(f"**Template:** {template_info['name']}")
                    st.write(f"**Description:** {template_info['description']}")
                    st.write("**Structure:**")
                    for i, step in enumerate(template_info['structure'], 1):
                        st.write(f"{i}. {step}")
            else:
                st.write("**Available Templates:**")
                templates = ["hero_journey", "romance", "mystery", "coming_of_age"]
                for template in templates:
                    template_info = load_story_template(template)
                    if template_info:
                        st.write(f"• **{template_info['name']}**: {template_info['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Collaboration Tools
        with st.expander("👥 Collaboration Tools", expanded=True):
            st.markdown('<div class="collaboration-tools">', unsafe_allow_html=True)
            st.write("**Current Collaborators:**")
            if st.session_state.collaborators:
                for collab in st.session_state.collaborators:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{collab['email']}**")
                    with col2:
                        st.write(f"Role: {collab['role']}")
                    with col3:
                        if st.button(f"Remove", key=f"remove_{collab['email']}"):
                            st.session_state.collaborators.remove(collab)
                            st.success(f"Removed {collab['email']}")
                            st.rerun()
            else:
                st.info("No collaborators added yet.")
            
            st.write("**Share Current Story:**")
            if st.session_state.current_story is not None:
                story_url = f"http://localhost:8501?story_id={st.session_state.current_story_id}"
                st.code(story_url)
                st.info("Share this URL with your collaborators!")
            else:
                st.info("Generate a story first to get a shareable URL!")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info(" Enter your mood in the sidebar to get started!")
        
        # Show technical features even when no story is generated
        st.markdown("---")
        st.header(" Technical Features")
        
        # Export Options (disabled when no story)
        with st.expander("📤 Export Options", expanded=True):
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.info("Generate a story first to enable export options!")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.button("📄 Export as PDF", disabled=True)
            with col2:
                st.button("🎬 Export as Screenplay", disabled=True)
            with col3:
                st.button("📊 Export as JSON", disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Version History
        with st.expander("📚 Version History", expanded=True):
            st.markdown('<div class="version-history">', unsafe_allow_html=True)
            if st.session_state.story_history:
                for version in reversed(st.session_state.story_history[-5:]):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**Version {version['version_id']}** - {version['mood'].title()}")
                            st.write(f"Model: {version['model']}")
                        with col2:
                            st.write(f"**{version['timestamp'][:10]}**")
                        with col3:
                            if st.button(f"Restore v{version['version_id']}", key=f"restore_{version['version_id']}"):
                                st.session_state.current_story = version.copy()
                                st.success(f"Restored version {version['version_id']}!")
                                st.rerun()
            else:
                st.info("No saved versions yet. Generate your first story to see version history!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Template Library
        with st.expander(" Template Library", expanded=True):
            st.markdown('<div class="template-library">', unsafe_allow_html=True)
            st.write("**Available Templates:**")
            templates = ["hero_journey", "romance", "mystery", "coming_of_age"]
            for template in templates:
                template_info = load_story_template(template)
                if template_info:
                    st.write(f"• **{template_info['name']}**: {template_info['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Collaboration Tools
        with st.expander("👥 Collaboration Tools", expanded=True):
            st.markdown('<div class="collaboration-tools">', unsafe_allow_html=True)
            st.write("**Current Collaborators:**")
            if st.session_state.collaborators:
                for collab in st.session_state.collaborators:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{collab['email']}**")
                    with col2:
                        st.write(f"Role: {collab['role']}")
                    with col3:
                        if st.button(f"Remove", key=f"remove_{collab['email']}"):
                            st.session_state.collaborators.remove(collab)
                            st.success(f"Removed {collab['email']}")
                            st.rerun()
            else:
                st.info("No collaborators added yet.")
            
            st.write("**Share Current Story:**")
            st.info("Generate a story first to get a shareable URL!")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
