import streamlit as st
import random
import re
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Handle torch import with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Using simplified story generation.")

# Handle transformers import with fallback
try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        T5Tokenizer,
        T5ForConditionalGeneration,
        BartTokenizer,
        BartForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Using template-based story generation.")

class StoryGenerator:
    """Story generator with fallback to template-based generation"""
    
    def __init__(self):
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Initialize models only if available
        if TRANSFORMERS_AVAILABLE:
            self.story_generator = self._load_story_model()
            self.title_generator = self._load_title_model()
        else:
            self.story_generator = None
            self.title_generator = None
        
        # Story templates and structures
        self.genre_templates = self._create_genre_templates()
        self.character_archetypes = self._create_character_archetypes()
        self.plot_structures = self._create_plot_structures()
        self.dialogue_styles = self._create_dialogue_styles()
        
    @st.cache_resource
    def _load_story_model(_self):
        """Load story generation model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            # Try GPT-2 medium first (good for creative writing)
            tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return generator
            
        except Exception as e:
            st.warning(f"Story model loading failed: {e}")
            try:
                # Fallback to smaller model
                generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
                )
                return generator
            except:
                return None
    
    @st.cache_resource
    def _load_title_model(_self):
        """Load model for generating titles and summaries"""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            # Use BART for summarization and title generation
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            
            generator = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
            )
            
            return generator
            
        except Exception as e:
            st.warning(f"Title model loading failed: {e}")
            return None
    
    def generate_story(self, mood: str, genre: str = None, length: str = "medium") -> Dict[str, Any]:
        """Generate story with fallback to template-based generation"""
        if self.story_generator and TRANSFORMERS_AVAILABLE:
            return self._ai_story_generation(mood, genre, length)
        else:
            return self._template_based_story_generation(mood, genre, length)
    
    def _template_based_story_generation(self, mood: str, genre: str = None, length: str = "medium") -> Dict[str, Any]:
        """Generate story using templates when AI models aren't available"""
        genre = genre or "Drama"  # Default genre
        
        # Get genre template
        template = self.genre_templates.get(genre, self.genre_templates['Drama'])
        
        # Generate story based on mood and genre
        story = self._create_story_from_template(mood, template, length)
        
        return {
            'title': f"The {mood.capitalize()} {genre}",
            'story': story,
            'genre': genre,
            'mood': mood,
            'length': length,
            'method': 'template_based'
        }
    
    def _create_story_from_template(self, mood: str, template: Dict, length: str) -> str:
        """Create story using template structure"""
        # This is a simplified template-based story generator
        # You can expand this with more sophisticated logic
        
        story_parts = []
        
        # Introduction based on mood
        mood_intros = {
            'happy': f"In a world where joy seemed endless, there lived a character whose heart overflowed with happiness.",
            'sad': f"In the depths of melancholy, where shadows danced with sorrow, there existed a soul seeking light.",
            'excited': f"With energy crackling like lightning, a spirited adventurer prepared for an extraordinary journey.",
            'calm': f"In the peaceful embrace of tranquility, where time seemed to stand still, a story began to unfold.",
            'mysterious': f"Behind the veil of the unknown, where secrets whispered in the darkness, a mystery awaited discovery."
        }
        
        intro = mood_intros.get(mood.lower(), "Once upon a time, in a world not unlike our own, a remarkable tale began.")
        story_parts.append(intro)
        
        # Add plot structure
        for element in template['structure']:
            story_parts.append(f"\n{element}: {self._generate_plot_element(element, mood, template)}")
        
        # Conclusion
        conclusion = f"\nAnd so, the {mood.lower()} journey came to an end, leaving behind memories that would last forever."
        story_parts.append(conclusion)
        
        return " ".join(story_parts)
    
    def _generate_plot_element(self, element: str, mood: str, template: Dict) -> str:
        """Generate content for a plot element"""
        # Simple content generation based on element type
        if element == "Setup":
            return f"The stage was set for a {mood.lower()} adventure, with characters ready to embark on their journey."
        elif element == "Inciting Incident":
            return f"Suddenly, everything changed when an unexpected event shook the foundation of their {mood.lower()} world."
        elif element == "Rising Action":
            return f"Challenges mounted as they navigated through the complexities of their {mood.lower()} situation."
        elif element == "Climax":
            return f"At the peak of tension, the ultimate test of their {mood.lower()} resolve arrived."
        elif element == "Resolution":
            return f"Finally, they found their way to a {mood.lower()} resolution, stronger and wiser than before."
        else:
            return f"The story continued to unfold, revealing new aspects of their {mood.lower()} journey."
    
    def _create_genre_templates(self):
        """Create detailed templates for different genres"""
        return {
            'Comedy': {
                'structure': ['Setup', 'Inciting Incident', 'Rising Action', 'Climax', 'Resolution'],
                'elements': ['humor', 'misunderstandings', 'quirky characters', 'light conflict', 'happy ending'],
                'tone': 'lighthearted, funny, optimistic',
                'character_goals': 'finding happiness, solving problems, learning lessons',
                'conflict_types': ['miscommunication', 'social awkwardness', 'family drama', 'workplace chaos'],
                'themes': ['friendship', 'love conquers all', 'being yourself', 'second chances']
            },
            
            'Drama': {
                'structure': ['Exposition', 'Rising Action', 'Climax', 'Falling Action', 'Resolution'],
                'elements': ['emotional depth', 'character development', 'meaningful conflict', 'life lessons'],
                'tone': 'serious, emotional, contemplative',
                'character_goals': 'personal growth, healing, understanding',
                'conflict_types': ['family secrets', 'moral dilemmas', 'loss', 'identity crisis'],
                'themes': ['redemption', 'family bonds', 'coming of age', 'forgiveness', 'sacrifice']
            },
            
            'Thriller': {
                'structure': ['Hook', 'Investigation', 'Complications', 'Climax', 'Resolution'],
                'elements': ['suspense', 'mystery', 'danger', 'plot twists', 'high stakes'],
                'tone': 'tense, suspenseful, dark',
                'character_goals': 'survival, uncovering truth, stopping evil',
                'conflict_types': ['conspiracy', 'chase', 'betrayal', 'deadly game'],
                'themes': ['justice vs vengeance', 'trust', 'power corruption', 'survival']
            },
            
            'Romance': {
                'structure': ['Meet Cute', 'Attraction', 'Obstacles', 'Dark Moment', 'Happy Ending'],
                'elements': ['love story', 'emotional connection', 'relationship obstacles', 'chemistry'],
                'tone': 'romantic, emotional, hopeful',
                'character_goals': 'finding true love, overcoming fears, commitment',
                'conflict_types': ['class differences', 'past trauma', 'family opposition', 'career vs love'],
                'themes': ['true love', 'second chances', 'destiny', 'sacrifice for love']
            },
            
            'Horror': {
                'structure': ['Normal World', 'First Scare', 'Escalation', 'Final Confrontation', 'Resolution'],
                'elements': ['fear', 'supernatural', 'survival', 'dark atmosphere', 'unknown threat'],
                'tone': 'scary, intense, foreboding',
                'character_goals': 'survival, escape, defeating evil',
                'conflict_types': ['haunting', 'monster', 'curse', 'psychological terror'],
                'themes': ['good vs evil', 'facing fears', 'consequences', 'survival instinct']
            },
            
            'Adventure': {
                'structure': ['Call to Adventure', 'Departure', 'Trials', 'Revelation', 'Return'],
                'elements': ['quest', 'heroism', 'exploration', 'discovery', 'brave deeds'],
                'tone': 'exciting, adventurous, inspiring',
                'character_goals': 'completing quest, saving others, finding treasure',
                'conflict_types': ['dangerous journey', 'evil villain', 'natural disasters', 'tests of courage'],
                'themes': ['heroism', 'friendship', 'courage', 'discovery', 'growing up']
            },
            
            'Sci-Fi': {
                'structure': ['World Building', 'Inciting Incident', 'Investigation', 'Revelation', 'New Order'],
                'elements': ['technology', 'future setting', 'scientific concepts', 'exploration'],
                'tone': 'imaginative, thought-provoking, futuristic',
                'character_goals': 'understanding truth, saving humanity, exploration',
                'conflict_types': ['AI rebellion', 'alien contact', 'time paradox', 'dystopian society'],
                'themes': ['humanity vs technology', 'progress', 'identity', 'future consequences']
            }
        }
    
    def _create_character_archetypes(self):
        """Character archetypes for different genres"""
        return {
            'Comedy': {
                'protagonists': ['the everyman', 'the dreamer', 'the neurotic', 'the optimist'],
                'supporting': ['the best friend', 'the mentor', 'the rival', 'the love interest'],
                'traits': ['quirky', 'relatable', 'flawed but lovable', 'determined']
            },
            'Drama': {
                'protagonists': ['the survivor', 'the seeker', 'the guardian', 'the rebel'],
                'supporting': ['the wise elder', 'the catalyst', 'the mirror', 'the antagonist'],
                'traits': ['complex', 'emotionally deep', 'struggling', 'evolving']
            },
            'Thriller': {
                'protagonists': ['the investigator', 'the victim', 'the hunter', 'the witness'],
                'supporting': ['the ally', 'the suspect', 'the mastermind', 'the informant'],
                'traits': ['determined', 'resourceful', 'paranoid', 'brave']
            },
            'Romance': {
                'protagonists': ['the romantic', 'the cynic', 'the innocent', 'the wounded'],
                'supporting': ['the matchmaker', 'the obstacle', 'the ex', 'the confidant'],
                'traits': ['passionate', 'vulnerable', 'hopeful', 'transformative']
            }
        }
    
    def _create_plot_structures(self):
        """Detailed plot structures for story generation"""
        return {
            'three_act': {
                'act1': 0.25,  # Setup - 25%
                'act2': 0.50,  # Confrontation - 50%
                'act3': 0.25   # Resolution - 25%
            },
            'five_act': {
                'exposition': 0.15,
                'rising_action': 0.25,
                'climax': 0.20,
                'falling_action': 0.25,
                'resolution': 0.15
            }
        }
    
    def _create_dialogue_styles(self):
        """Dialogue styles for different genres"""
        return {
            'Comedy': {
                'style': 'witty, playful, conversational',
                'techniques': ['wordplay', 'misunderstandings', 'comedic timing', 'pop culture references']
            },
            'Drama': {
                'style': 'emotional, realistic, introspective',
                'techniques': ['subtext', 'emotional revelation', 'conflict', 'meaningful pauses']
            },
            'Thriller': {
                'style': 'tense, urgent, mysterious',
                'techniques': ['short sentences', 'cryptic hints', 'pressure', 'revelation']
            }
        }
    
    def _generate_with_llm(self, prompt: str, creative_mode: bool = False) -> str:
        """Generate story using local LLM"""
        if not TRANSFORMERS_AVAILABLE or not self.story_generator:
            st.warning("LLM generation is not available. Using template-based generation.")
            return self._generate_with_templates(prompt, creative_mode) # Fallback to template-based

        try:
            # Configure generation parameters
            generation_kwargs = {
                'max_new_tokens': 800 if creative_mode else 600,
                'temperature': 0.9 if creative_mode else 0.8,
                'do_sample': True,
                'top_p': 0.95,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'pad_token_id': self.story_generator.tokenizer.eos_token_id
            }
            
            # Generate story
            result = self.story_generator(
                prompt,
                **generation_kwargs,
                return_full_text=False
            )
            
            generated_text = result[0]['generated_text'] if result else ""
            
            # Clean up the generated text
            return self._clean_generated_text(generated_text)
            
        except Exception as e:
            st.warning(f"LLM generation failed: {e}")
            return ""
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean and format generated text"""
        # Remove the original prompt if it's repeated
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Write a') and not line.startswith('STORY REQUIREMENTS'):
                cleaned_lines.append(line)
        
        return '\n\n'.join(cleaned_lines)
    
    def _generate_with_templates(self, emotion: str, genre: str, intensity: int, 
                               length: str, template: Dict[str, Any]) -> str:
        """Fallback template-based generation"""
        
        scene_count = self._get_scene_count(length)
        
        # Get template stories based on emotion and genre
        story_templates = self._get_template_stories()
        template_key = f"{emotion}_{genre}".lower()
        
        if template_key not in story_templates:
            template_key = f"{emotion}_drama"
        if template_key not in story_templates:
            template_key = "joy_comedy"
        
        base_story = story_templates[template_key]
        
        # Adapt length
        scenes = base_story['scenes'][:scene_count]
        
        # Format as script
        script_parts = []
        for i, scene in enumerate(scenes, 1):
            script_parts.append(f"SCENE {i}:\n{scene}")
        
        return '\n\n'.join(script_parts)
    
    def _get_template_stories(self) -> Dict[str, Dict[str, Any]]:
        """Pre-written template stories for fallback"""
        return {
            'joy_comedy': {
                'scenes': [
                    "INT. COFFEE SHOP - MORNING\n\nSAM (20s) spills coffee on their laptop, panicking about a presentation. A STRANGER (20s) offers help with a smile.\n\nSTRANGER\nHappens to the best of us. I'm a tech repair wizard.\n\nSAM\n(relieved)\nYou're a lifesaver!",
                    
                    "INT. TECH REPAIR SHOP - LATER\n\nThe Stranger, ALEX, works on Sam's laptop. They share stories and laugh.\n\nALEX\nYou know, disasters make the best stories later.\n\nSAM\nIf that's true, I'm writing a bestseller.",
                    
                    "INT. OFFICE BUILDING - DAY\n\nSam delivers a flawless presentation, grateful for the morning's 'disaster.' Alex watches from the audience, having been invited.\n\nSAM\n(to audience)\nSometimes the best opportunities come from the worst moments.\n\nAlex smiles knowingly from the back row.",
                    
                    "EXT. COFFEE SHOP - EVENING\n\nSam and Alex meet again at the same coffee shop, this time intentionally.\n\nALEX\nSame time tomorrow? I promise not to fix your laptop this time.\n\nSAM\n(laughing)\nDeal. But maybe bring a towel, just in case.\n\nThey clink coffee cups as the sun sets."
                ]
            },
            
            'sadness_drama': {
                'scenes': [
                    "INT. MAYA'S APARTMENT - NIGHT\n\nMAYA (30s) sits alone, surrounded by boxes. She finds an old photo of her grandmother. Tears fall as memories flood back.\n\nMAYA\n(whispered)\nI miss you, Grandma.",
                    
                    "INT. GRANDMOTHER'S HOUSE - FLASHBACK - DAY\n\nYoung Maya (8) bakes cookies with GRANDMOTHER (70s). The kitchen is warm and filled with laughter.\n\nGRANDMOTHER\nRemember, Maya, love lives in the little moments.\n\nYOUNG MAYA\nWhat do you mean?\n\nGRANDMOTHER\nYou'll understand when you need to.",
                    
                    "INT. MAYA'S APARTMENT - CONTINUOUS\n\nMaya finds a recipe card in Grandmother's handwriting. She starts gathering ingredients, finding comfort in the familiar routine.\n\nMAYA\n(to photo)\nI understand now. You're in every little moment.",
                    
                    "INT. MAYA'S KITCHEN - LATER\n\nMaya takes fresh cookies from the oven. The apartment smells like love. She packages some cookies and heads to the door.\n\nMAYA\n(voice-over)\nLove doesn't end. It just finds new ways to show up."
                ]
            },
            
            'anger_thriller': {
                'scenes': [
                    "INT. ALEX'S OFFICE - NIGHT\n\nALEX (30s) discovers incriminating documents about their company's illegal activities. Security cameras record everything.\n\nALEX\n(into phone)\nWe need to talk. Now.",
                    
                    "EXT. PARKING GARAGE - NIGHT\n\nAlex meets JORDAN (40s), a journalist. Footsteps echo ominously.\n\nJORDAN\nYou sure about this? These people don't forgive.\n\nALEX\nI can't stay silent anymore.",
                    
                    "INT. ALEX'S CAR - MOVING - NIGHT\n\nAlex notices they're being followed. Heart racing, they take evasive action through city streets.\n\nALEX\n(panicked)\nThey know. They already know!",
                    
                    "INT. NEWS STATION - DAWN\n\nAlex and Jordan prepare for a live broadcast. Despite the danger, Alex is determined.\n\nALEX\n(to camera)\nSometimes the right thing to do is also the hardest thing to do.\n\nThe red light goes on. Truth time."
                ]
            },
            
            'fear_horror': {
                'scenes': [
                    "INT. OLD HOUSE - NIGHT\n\nSARAH (20s) enters the inherited house. Creaking floors and shadows everywhere. Her phone has no signal.\n\nSARAH\n(nervous)\nJust pack and leave. Simple.",
                    
                    "INT. ATTIC - CONTINUOUS\n\nSarah finds old family photos. In each one, a dark figure appears in the background, getting closer in each successive photo.\n\nSARAH\n(horrified)\nThat's... that's not possible.",
                    
                    "INT. HALLWAY - NIGHT\n\nSarah tries to leave but doors slam shut. Whispers fill the air. She's trapped with whatever haunts this place.\n\nWHISPERS\n(echoing)\nStay... stay with us...",
                    
                    "INT. LIVING ROOM - DAWN\n\nSarah confronts the spirit, her fear transformed into understanding.\n\nSARAH\nI'm not afraid anymore. You're just lost, aren't you?\n\nThe house grows quiet. Sunlight streams through windows."
                ]
            },
            
            'love_romance': {
                'scenes': [
                    "INT. BOOKSTORE - DAY\n\nEMILY (25) reaches for the same book as DAVID (27). Their hands touch. Time seems to stop.\n\nDAVID\nGreat taste in literature.\n\nEMILY\n(blushing)\nYou can have it. I've read it three times already.",
                    
                    "EXT. PARK BENCH - SUNSET\n\nEmily and David sit together, the book between them. They share stories and dreams.\n\nDAVID\nI never believed in love at first sight.\n\nEMILY\nAnd now?\n\nDAVID\nNow I believe in love at first touch.",
                    
                    "INT. EMILY'S APARTMENT - EVENING\n\nEmily discovers David is moving across the country tomorrow. Her world crumbles.\n\nEMILY\n(tearful)\nWhy didn't you tell me?\n\nDAVID\nBecause I knew it would hurt this much.",
                    
                    "INT. AIRPORT - DAY\n\nEmily rushes through the terminal. She finds David at his gate.\n\nEMILY\n(breathless)\nDon't go. Or... take me with you.\n\nDAVID\n(amazed)\nAre you sure?\n\nEMILY\nI've never been more sure of anything.\n\nThey embrace as his flight is called."
                ]
            }
        }
    
    def _get_scene_count(self, length: str) -> int:
        """Convert length description to scene count"""
        if "Short" in length:
            return 4
        elif "Medium" in length:
            return 6
        elif "Long" in length:
            return 8
        return 4
    
    def _parse_and_enhance_story(self, story_content: str, genre: str, template: Dict[str, Any]) -> str:
        """Parse and enhance the generated story"""
        if not story_content.strip():
            return "Story generation failed. Please try again."
        
        # Basic formatting
        lines = story_content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Format scene headers
            if line.startswith('SCENE') or line.startswith('INT.') or line.startswith('EXT.'):
                enhanced_lines.append(f"\n{line.upper()}")
            # Format character names
            elif line.isupper() and len(line.split()) <= 3 and line.isalpha():
                enhanced_lines.append(f"\n{line}")
            else:
                enhanced_lines.append(line)
        
        enhanced_story = '\n'.join(enhanced_lines)
        
        # Add genre-specific enhancements
        if genre == 'Comedy':
            enhanced_story = self._add_comedy_beats(enhanced_story)
        elif genre == 'Thriller':
            enhanced_story = self._add_suspense_beats(enhanced_story)
        
        return enhanced_story
    
    def _add_comedy_beats(self, script: str) -> str:
        """Add comedy timing and beats"""
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        lines = script.split('\n')
        enhanced = []
        
        for line in lines:
            enhanced.append(line)
            # Add pause after potential punchlines
            if '?' in line and any(word in line.lower() for word in ['what', 'how', 'why']):
                enhanced.append("(beat)")
        
        return '\n'.join(enhanced)
    
    def _add_suspense_beats(self, script: str) -> str:
        """Add suspense and tension beats"""
        lines = script.split('\n')
        enhanced = []
        
        for line in lines:
            enhanced.append(line)
            # Add tension beats
            if any(word in line.lower() for word in ['footsteps', 'shadow', 'noise', 'door']):
                enhanced.append("(pause, listening)")
        
        return '\n'.join(enhanced)
    
    def _generate_title_and_tagline(self, script: str, genre: str, emotion: str) -> tuple:
        """Generate title and tagline"""
        
        # Extract key words from script for title inspiration
        words = re.findall(r'\b[A-Z][a-z]+\b', script)
        common_words = ['The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By']
        meaningful_words = [word for word in words if word not in common_words][:10]
        
        # Genre-specific title patterns
        title_patterns = {
            'Comedy': [
                f"The {random.choice(meaningful_words) if meaningful_words else 'Comedy'} Chronicles",
                f"{random.choice(['When', 'How', 'Why'])} {emotion.title()} Met {genre}",
                f"The Great {emotion.title()} Adventure"
            ],
            'Drama': [
                f"Letters to {random.choice(meaningful_words) if meaningful_words else 'Tomorrow'}",
                f"The {emotion.title()} Within",
                f"Echoes of {random.choice(meaningful_words) if meaningful_words else 'Memory'}"
            ],
            'Thriller': [
                f"The {random.choice(meaningful_words) if meaningful_words else 'Truth'} Conspiracy",
                f"Dark {random.choice(meaningful_words) if meaningful_words else 'Secrets'}",
                f"Final {random.choice(meaningful_words) if meaningful_words else 'Hour'}"
            ],
            'Romance': [
                f"Love in the Time of {emotion.title()}",
                f"The {random.choice(meaningful_words) if meaningful_words else 'Heart'} Waltz",
                f"Midnight {random.choice(meaningful_words) if meaningful_words else 'Promise'}"
            ],
            'Horror': [
                f"The {random.choice(meaningful_words) if meaningful_words else 'Haunting'} House",
                f"Whispers of {random.choice(meaningful_words) if meaningful_words else 'Darkness'}",
                f"The Last {random.choice(meaningful_words) if meaningful_words else 'Hope'}"
            ],
            'Adventure': [
                f"Quest for {random.choice(meaningful_words) if meaningful_words else 'Truth'}",
                f"The {random.choice(meaningful_words) if meaningful_words else 'Lost'} Expedition",
                f"Beyond {random.choice(meaningful_words) if meaningful_words else 'Tomorrow'}"
            ],
            'Sci-Fi': [
                f"The {random.choice(meaningful_words) if meaningful_words else 'Future'} Protocol",
                f"Digital {random.choice(meaningful_words) if meaningful_words else 'Dreams'}",
                f"Tomorrow's {random.choice(meaningful_words) if meaningful_words else 'Child'}"
            ]
        }
        
        # Select title
        patterns = title_patterns.get(genre, title_patterns['Drama'])
        title = random.choice(patterns)
        
        # Generate taglines
        tagline_patterns = {
            'Comedy': [
                "Sometimes the best days come after the worst mistakes",
                "Laughter is the best medicine, but timing is everything",
                "Life's too short to take seriously"
            ],
            'Drama': [
                "Every ending is a new beginning",
                "The heart knows what the mind cannot understand",
                "Some stories can only be told through tears"
            ],
            'Thriller': [
                "The truth has a way of surfacing",
                "Trust no one, suspect everyone",
                "Some secrets are worth killing for"
            ],
            'Romance': [
                "Love finds a way",
                "Sometimes the heart knows first",
                "True love never gives up"
            ],
            'Horror': [
                "Fear has a new address",
                "Some doors should never be opened",
                "The past never stays buried"
            ],
            'Adventure': [
                "Every journey begins with a single step",
                "Fortune favors the bold",
                "The greatest treasures are found within"
            ],
            'Sci-Fi': [
                "The future is what we make it",
                "Progress comes with a price",
                "Humanity's greatest test awaits"
            ]
        }
        
        taglines = tagline_patterns.get(genre, tagline_patterns['Drama'])
        tagline = random.choice(taglines)
        
        return title, tagline
    
def _create_story_summary(self, script: str, genre: str, themes: List[str]) -> str:
    """Create a compelling story summary"""
    
    # Extract character names from script
    character_names = re.findall(r'^([A-Z][A-Z\s]*[A-Z])', script, re.MULTILINE)
    main_character = character_names[0] if character_names else "the protagonist"
    
    # Create theme-based summary templates
    summary_templates = {
        'Comedy': f"{main_character} learns that sometimes the best adventures begin with the biggest mistakes, discovering that {themes[0] if themes else 'friendship'} can turn any disaster into a triumph.",
        'Drama': f"When {main_character} faces a life-changing moment, they must confront their past and discover that {themes[0] if themes else 'healing'} comes from the most unexpected places.",
        'Thriller': f"{main_character} uncovers a dangerous truth that puts everything they care about at risk, forcing them to choose between safety and {themes[0] if themes else 'justice'}.",
        'Romance': f"Against all odds, {main_character} discovers that love isn't just about finding the right person, but about the courage to fight for {themes[0] if themes else 'true connection'}.",
        'Horror': f"When {main_character} confronts an ancient evil, they must overcome their deepest fears to protect those they love and restore {themes[0] if themes else 'peace'}.",
        'Adventure': f"On an epic quest, {main_character} discovers that the greatest treasures aren't gold or glory, but the {themes[0] if themes else 'courage'} found within themselves.",
        'Sci-Fi': f"In a world transformed by technology, {main_character} must navigate the fine line between progress and humanity, ultimately learning that {themes[0] if themes else 'connection'} transcends all boundaries."
    }
    
    return summary_templates.get(genre, summary_templates['Drama'])

def generate_alternative_endings(self, story_data: Dict[str, str], count: int = 3) -> List[str]:
    """Generate alternative endings based on story data"""
    
    base_story = story_data.get('script', '')
    genre = story_data.get('genre', 'Drama')
    
    # Create different ending approaches
    ending_types = {
        'happy': 'All conflicts resolve positively, characters achieve their goals',
        'bittersweet': 'Victory comes with sacrifice, mixed emotions',
        'twist': 'Unexpected revelation changes everything we thought we knew',
        'open': 'Questions remain, future possibilities suggested',
        'dark': 'Consequences of actions lead to sobering realizations'
    }
    
    endings = []
    selected_types = list(ending_types.keys())[:count]
    
    for ending_type in selected_types:
        prompt = f"""
        Create an alternative {ending_type} ending for this {genre} story:
        
        Story context: {base_story[-500:] if len(base_story) > 500 else base_story}
        
        Ending style: {ending_types[ending_type]}
        
        Write a compelling 2-3 paragraph ending that fits this style:
        """
        
        try:
            # Assuming _generate_with_openai is a placeholder for a different LLM call
            # For now, we'll use a simple template-based approach or a placeholder
            # If _generate_with_openai is not defined, this will cause an error.
            # For the purpose of this edit, we'll assume it's a placeholder.
            # In a real scenario, this would involve an external API call.
            # For now, we'll just return a placeholder.
            endings.append({
                'type': ending_type.title(),
                'content': f"A {ending_type} conclusion where the story reaches a meaningful resolution."
            })
        except Exception as e:
            st.error(f"Error generating {ending_type} ending: {str(e)}")
            endings.append({
                'type': ending_type.title(),
                'content': f"A {ending_type} conclusion where the story reaches a meaningful resolution."
            })
    
    return endings
    
    def get_story_metrics(self, script: str) -> Dict[str, Any]:
        """Calculate story metrics and statistics"""
        if not script:
            return {}
        
        # Basic metrics
        words = script.split()
        lines = script.split('\n')
        
        # Count scenes
        scene_count = len([line for line in lines if 'SCENE' in line.upper() or line.startswith(('INT.', 'EXT.'))])
        
        # Count dialogue vs description
        dialogue_lines = len([line for line in lines if line.isupper() and line.strip() and len(line.split()) <= 3])
        action_lines = len([line for line in lines if line.strip() and not line.isupper() and not line.startswith(('SCENE', 'INT.', 'EXT.'))])
        
        # Calculate ratios
        total_lines = len([line for line in lines if line.strip()])
        dialogue_ratio = dialogue_lines / total_lines if total_lines > 0 else 0
        
        # Estimate timing (250 words per minute for reading, 1 page = ~1 minute screen time)
        reading_time = len(words) / 250
        screen_time = len(words) / 250  # Rough estimate
        
        return {
            'word_count': len(words),
            'line_count': len(lines),
            'estimated_scenes': max(1, scene_count),
            'dialogue_lines': dialogue_lines,
            'action_lines': action_lines,
            'dialogue_ratio': dialogue_ratio,
            'reading_time_minutes': round(reading_time, 1),
            'estimated_screen_time_minutes': round(screen_time, 1)
        }