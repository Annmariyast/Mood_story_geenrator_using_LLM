import random
import re
from typing import List, Dict, Tuple

def get_soundtrack_recommendation(mood: str, genre: str = None) -> Dict[str, List[str]]:
    """
    Generate soundtrack recommendations based on mood and genre
    
    Args:
        mood (str): The detected mood
        genre (str): The movie genre (optional)
    
    Returns:
        Dict containing soundtrack recommendations
    """
    # Mood-based soundtrack mappings
    mood_soundtracks = {
        "happy": [
            "Upbeat pop songs", "Feel-good rock", "Cheerful electronic",
            "Motivational hip-hop", "Joyful folk music"
        ],
        "sad": [
            "Melancholic ballads", "Emotional piano pieces", "Soulful blues",
            "Heartfelt acoustic", "Touching orchestral"
        ],
        "excited": [
            "High-energy electronic", "Pumping rock anthems", "Dynamic orchestral",
            "Fast-paced pop", "Energetic hip-hop"
        ],
        "calm": [
            "Ambient soundscapes", "Gentle acoustic", "Peaceful classical",
            "Smooth jazz", "Relaxing nature sounds"
        ],
        "mysterious": [
            "Atmospheric electronic", "Dark ambient", "Suspenseful orchestral",
            "Mystical world music", "Haunting melodies"
        ],
        "romantic": [
            "Love ballads", "Soft jazz", "Tender acoustic",
            "Emotional pop", "Romantic classical"
        ],
        "tense": [
            "Suspenseful orchestral", "Dark electronic", "Intense rock",
            "Dramatic classical", "Thrilling soundtracks"
        ]
    }
    
    # Genre-specific additions
    genre_soundtracks = {
        "action": ["Epic orchestral", "High-energy rock", "Pumping electronic"],
        "drama": ["Emotional orchestral", "Deep acoustic", "Soulful jazz"],
        "comedy": ["Light-hearted pop", "Funky grooves", "Cheerful folk"],
        "horror": ["Dark ambient", "Creepy sound effects", "Tense orchestral"],
        "sci-fi": ["Futuristic electronic", "Space ambient", "Synthetic sounds"]
    }
    
    # Get base recommendations for mood
    base_recommendations = mood_soundtracks.get(mood.lower(), ["Versatile soundtrack"])
    
    # Add genre-specific recommendations if available
    if genre and genre.lower() in genre_soundtracks:
        base_recommendations.extend(genre_soundtracks[genre.lower()])
    
    return {
        "mood_based": base_recommendations[:3],
        "genre_specific": genre_soundtracks.get(genre.lower(), []) if genre else [],
        "overall_style": f"{mood.capitalize()} and {genre.lower() if genre else 'versatile'} soundtrack"
    }

def format_script(script_text: str) -> str:
    """
    Format a movie script with proper structure and styling
    
    Args:
        script_text (str): Raw script text
    
    Returns:
        Formatted script text
    """
    if not script_text:
        return "No script available."
    
    # Basic script formatting
    formatted = script_text.strip()
    
    # Add scene headers if they don't exist
    if "SCENE" not in formatted.upper() and "INT." not in formatted.upper() and "EXT." not in formatted.upper():
        formatted = f"SCENE 1\n{formatted}"
    
    # Add basic structure if missing
    if "FADE IN" not in formatted.upper():
        formatted = "FADE IN\n\n" + formatted
    
    if "FADE OUT" not in formatted.upper():
        formatted = formatted + "\n\nFADE OUT"
    
    return formatted

def get_genre_icons() -> Dict[str, str]:
    """
    Get emoji icons for different movie genres
    
    Returns:
        Dict mapping genre names to emoji icons
    """
    return {
        "action": "💥",
        "adventure": "🧗‍♂️",
        "comedy": "🎭",
        "drama": "🎭",
        "horror": "🎭",
        "romance": "💕",
        "sci-fi": "🚀",
        "thriller": "🎭",
        "fantasy": "🧗‍♂️",
        "mystery": "🕵️",
        "crime": "🕵️",
        "war": "⚔️",
        "western": "🎭",
        "musical": "🎵",
        "documentary": "📹",
        "animation": "🎨",
        "family": "👩‍👧‍👦",
        "biography": "🎭",
        "history": "🏛️",
        "sport": "⚽"
    }

def get_mood_emoji(mood: str) -> str:
    """
    Get emoji representation for different moods
    
    Args:
        mood (str): The mood name
    
    Returns:
        Emoji string for the mood
    """
    mood_emojis = {
        "happy": "🎉",
        "sad": "😢",
        "excited": "🤩",
        "calm": "😌",
        "mysterious": "🤔",
        "romantic": "💕",
        "tense": "😰",
        "angry": "😠",
        "surprised": "😲",
        "confused": "😕",
        "confident": "🤔",
        "nervous": "😰"
    }
    return mood_emojis.get(mood.lower(), "🎭")

def clean_text(text: str) -> str:
    """
    Clean and normalize text input
    
    Args:
        text (str): Raw text input
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
    
    return cleaned

def validate_mood_input(text: str) -> Tuple[bool, str]:
    """
    Validate mood input text
    
    Args:
        text (str): Input text to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or len(text.strip()) < 3:
        return False, "Please enter at least 3 characters to describe your mood."
    
    if len(text.strip()) > 500:
        return False, "Please keep your mood description under 500 characters."
    
    return True, ""

def get_random_quote() -> str:
    """
    Get a random inspirational quote for the app
    
    Returns:
        Random quote string
    """
    quotes = [
        "Every mood tells a story. Let's find yours.",
        "Transform your feelings into cinematic magic.",
        "Your emotions are the script, AI is the director.",
        "From mood to masterpiece in moments.",
        "Let your heart write the story.",
        "Emotions are the colors of life's canvas.",
        "Turn your mood into a movie masterpiece.",
        "Every feeling deserves its own story."
    ]
    return random.choice(quotes)
