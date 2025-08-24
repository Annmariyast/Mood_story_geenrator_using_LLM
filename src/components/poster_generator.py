import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
import base64
from typing import Optional, Dict, Any, Tuple
import random
import math
import colorsys

# Handle torch import with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch not available. Using template-based poster generation.")

# Handle diffusers import with fallback
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    st.warning("⚠️ Diffusers not available. Using template-based poster generation.")

class PosterGenerator:
    """Poster generator with fallback to template-based generation"""
    
    def __init__(self):
        self.poster_sizes = {
            'standard': (400, 600),
            'wide': (600, 400),
            'square': (500, 500)
        }
        
        self.color_schemes = self._create_color_schemes()
        self.design_templates = self._create_design_templates()
        self.font_fallbacks = self._get_font_fallbacks()
        
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # Initialize models only if available
        if DIFFUSERS_AVAILABLE and TORCH_AVAILABLE:
            self.poster_generator = self._load_poster_model()
        else:
            self.poster_generator = None
        
        # Poster templates and styles
        self.poster_templates = self._create_poster_templates()
        self.color_schemes = self._create_color_schemes()
        self.typography_styles = self._create_typography_styles()
        
    def _create_color_schemes(self):
        """Color schemes for different genres"""
        return {
            'Comedy': {
                'primary': (255, 107, 107),      # Bright red
                'secondary': (78, 205, 196),     # Turquoise
                'accent': (255, 230, 109),       # Yellow
                'background': (255, 248, 225),   # Cream
                'text': (51, 51, 51)             # Dark gray
            },
            'Drama': {
                'primary': (141, 110, 99),       # Brown
                'secondary': (161, 136, 127),    # Light brown
                'accent': (215, 204, 200),       # Beige
                'background': (245, 245, 220),   # Beige
                'text': (62, 39, 35)             # Dark brown
            },
            'Thriller': {
                'primary': (38, 50, 56),         # Dark blue-gray
                'secondary': (69, 90, 100),      # Blue-gray
                'accent': (255, 82, 82),         # Red
                'background': (236, 239, 241),   # Light gray
                'text': (33, 33, 33)             # Almost black
            },
            'Romance': {
                'primary': (233, 30, 99),        # Pink
                'secondary': (248, 187, 208),    # Light pink
                'accent': (252, 228, 236),       # Very light pink
                'background': (255, 240, 245),   # Lavender blush
                'text': (136, 14, 79)            # Dark pink
            },
            'Horror': {
                'primary': (183, 28, 28),        # Dark red
                'secondary': (66, 66, 66),       # Dark gray
                'accent': (255, 138, 128),       # Light red
                'background': (33, 33, 33),      # Very dark gray
                'text': (255, 255, 255)          # White
            },
            'Adventure': {
                'primary': (46, 125, 50),        # Green
                'secondary': (102, 187, 106),    # Light green
                'accent': (165, 214, 167),       # Very light green
                'background': (232, 245, 233),   # Light green
                'text': (27, 94, 32)             # Dark green
            },
            'Sci-Fi': {
                'primary': (21, 101, 192),       # Blue
                'secondary': (66, 165, 245),     # Light blue
                'accent': (187, 222, 251),       # Very light blue
                'background': (227, 242, 253),   # Light blue
                'text': (13, 71, 161)            # Dark blue
            },
            'Fantasy': {
                'primary': (123, 31, 162),       # Purple
                'secondary': (149, 117, 205),    # Light purple
                'accent': (209, 196, 233),       # Very light purple
                'background': (243, 229, 245),   # Light purple
                'text': (74, 20, 140)            # Dark purple
            }
        }
    
    def _create_design_templates(self):
        """Design templates for different poster styles"""
        return {
            'minimalist': {
                'title_size_ratio': 0.12,
                'tagline_size_ratio': 0.04,
                'title_position': (0.5, 0.8),
                'tagline_position': (0.5, 0.9),
                'design_elements': ['gradient_background', 'simple_shapes']
            },
            'classic': {
                'title_size_ratio': 0.15,
                'tagline_size_ratio': 0.05,
                'title_position': (0.5, 0.15),
                'tagline_position': (0.5, 0.25),
                'design_elements': ['frame_border', 'decorative_elements']
            },
            'modern': {
                'title_size_ratio': 0.18,
                'tagline_size_ratio': 0.06,
                'title_position': (0.5, 0.7),
                'tagline_position': (0.5, 0.85),
                'design_elements': ['geometric_shapes', 'bold_colors']
            },
            'artistic': {
                'title_size_ratio': 0.14,
                'tagline_size_ratio': 0.05,
                'title_position': (0.3, 0.6),
                'tagline_position': (0.3, 0.75),
                'design_elements': ['abstract_shapes', 'artistic_effects']
            }
        }
    
    def _get_font_fallbacks(self):
        """Get available font fallbacks"""
        return [
            'Arial',
            'Helvetica',
            'Times New Roman',
            'Courier New',
            'Verdana',
            'DejaVu Sans',
            'Liberation Sans'
        ]
    
    @st.cache_resource
    def _load_poster_model(_self):
        """Load poster generation model"""
        if not DIFFUSERS_AVAILABLE or not TORCH_AVAILABLE:
            return None
            
        try:
            # Load Stable Diffusion for image generation
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if TORCH_AVAILABLE else torch.float32
            )
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                pipe = pipe.to("cuda")
            
            return pipe
            
        except Exception as e:
            st.warning(f"Poster model loading failed: {e}")
            return None
    
    def generate_poster(self, title: str, genre: str, mood: str) -> Dict[str, Any]:
        """Generate poster with fallback to template-based generation"""
        if self.poster_generator and DIFFUSERS_AVAILABLE:
            return self._ai_poster_generation(title, genre, mood)
        else:
            return self._template_based_poster_generation(title, genre, mood)
    
    def _template_based_poster_generation(self, title: str, genre: str, mood: str) -> Dict[str, Any]:
        """Generate poster using templates when AI models aren't available"""
        # Get appropriate template
        template = self.poster_templates.get(genre, self.poster_templates['Drama'])
        color_scheme = self.color_schemes.get(mood, self.color_schemes['neutral'])
        
        # Generate poster description
        poster_description = self._create_poster_description(title, genre, mood, template, color_scheme)
        
        return {
            'title': title,
            'genre': genre,
            'mood': mood,
            'poster_description': poster_description,
            'color_scheme': color_scheme,
            'method': 'template_based',
            'image_url': None  # No AI-generated image
        }
    
    def _create_poster_description(self, title: str, genre: str, mood: str, template: Dict, color_scheme: Dict) -> str:
        """Create detailed poster description using templates"""
        description_parts = []
        
        # Main visual elements
        description_parts.append(f"A cinematic poster featuring the title '{title}' prominently displayed.")
        description_parts.append(f"The poster follows a {genre.lower()} style with {mood.lower()} undertones.")
        
        # Color scheme
        description_parts.append(f"Color palette: {', '.join(color_scheme['colors'])}")
        
        # Layout elements
        if template.get('layout'):
            description_parts.append(f"Layout: {template['layout']}")
        
        # Mood-specific elements
        mood_elements = {
            'happy': "Bright lighting, warm tones, uplifting composition",
            'sad': "Soft shadows, cool tones, contemplative atmosphere",
            'excited': "Dynamic angles, vibrant colors, energetic composition",
            'calm': "Smooth lines, gentle gradients, peaceful arrangement",
            'mysterious': "Dark shadows, contrasting elements, enigmatic atmosphere"
        }
        
        mood_desc = mood_elements.get(mood.lower(), "Balanced composition with appropriate mood elements")
        description_parts.append(f"Mood elements: {mood_desc}")
        
        # Genre-specific details
        if template.get('visual_elements'):
            description_parts.append(f"Genre elements: {', '.join(template['visual_elements'])}")
        
        return " ".join(description_parts)
    
    def _create_base_poster(self, colors: Dict[str, tuple], template: Dict[str, Any], genre: str) -> Image.Image:
        """Create the base poster image"""
        
        width, height = self.poster_sizes['standard']
        
        # Create base image
        image = Image.new('RGB', (width, height), colors['background'])
        draw = ImageDraw.Draw(image)
        
        # Create background based on genre
        if genre == 'Horror':
            # Dark, ominous background
            image = self._create_dark_gradient(image, colors)
        elif genre == 'Romance':
            # Soft, romantic background
            image = self._create_romantic_background(image, colors)
        elif genre == 'Sci-Fi':
            # Futuristic background
            image = self._create_scifi_background(image, colors)
        elif genre == 'Adventure':
            # Dynamic background
            image = self._create_adventure_background(image, colors)
        elif genre == 'Comedy':
            # Bright, cheerful background
            image = self._create_comedy_background(image, colors)
        else:
            # Default gradient background
            image = self._create_gradient_background(image, colors)
        
        return image
    
    def _create_gradient_background(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a gradient background"""
        width, height = image.size
        
        # Create gradient from top to bottom
        for y in range(height):
            # Calculate color interpolation
            ratio = y / height
            
            # Interpolate between primary and secondary colors
            r = int(colors['primary'][0] * (1 - ratio) + colors['secondary'][0] * ratio)
            g = int(colors['primary'][1] * (1 - ratio) + colors['secondary'][1] * ratio)
            b = int(colors['primary'][2] * (1 - ratio) + colors['secondary'][2] * ratio)
            
            # Draw horizontal line
            draw = ImageDraw.Draw(image)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return image
    
    def _create_dark_gradient(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a dark, ominous gradient for horror"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Dark gradient with some noise
        for y in range(height):
            ratio = y / height
            noise = random.randint(-20, 20)
            
            r = int(colors['background'][0] * (1 - ratio) + colors['primary'][0] * ratio) + noise
            g = int(colors['background'][1] * (1 - ratio) + colors['primary'][1] * ratio) + noise
            b = int(colors['background'][2] * (1 - ratio) + colors['primary'][2] * ratio) + noise
            
            # Clamp values
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return image
    
    def _create_romantic_background(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a soft, romantic background"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Soft radial gradient
        center_x, center_y = width // 2, height // 3
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for x in range(width):
            for y in range(height):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(1.0, distance / max_distance)
                
                r = int(colors['accent'][0] * (1 - ratio) + colors['secondary'][0] * ratio)
                g = int(colors['accent'][1] * (1 - ratio) + colors['secondary'][1] * ratio)
                b = int(colors['accent'][2] * (1 - ratio) + colors['secondary'][2] * ratio)
                
                if x % 4 == 0 and y % 4 == 0:  # Skip pixels for performance
                    draw.rectangle([(x, y), (x+3, y+3)], fill=(r, g, b))
        
        return image
    
    def _create_scifi_background(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a futuristic sci-fi background"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Create tech-like grid pattern
        grid_size = 20
        
        # Base gradient
        image = self._create_gradient_background(image, colors)
        
        # Add grid lines
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=colors['accent'], width=1)
        
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=colors['accent'], width=1)
        
        # Add some "circuit" elements
        for _ in range(10):
            x, y = random.randint(0, width-50), random.randint(0, height-50)
            draw.rectangle([(x, y), (x+30, y+5)], fill=colors['primary'])
            draw.rectangle([(x+10, y-10), (x+15, y+15)], fill=colors['secondary'])
        
        return image
    
    def _create_adventure_background(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a dynamic adventure background"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Dynamic diagonal gradient
        for x in range(width):
            for y in range(height):
                ratio = (x + y) / (width + height)
                
                r = int(colors['primary'][0] * (1 - ratio) + colors['secondary'][0] * ratio)
                g = int(colors['primary'][1] * (1 - ratio) + colors['secondary'][1] * ratio)
                b = int(colors['primary'][2] * (1 - ratio) + colors['secondary'][2] * ratio)
                
                if (x + y) % 6 == 0:  # Sample every 6th pixel for performance
                    draw.rectangle([(x, y), (x+5, y+5)], fill=(r, g, b))
        
        # Add some adventure elements (mountains/peaks)
        peak_points = []
        for i in range(5):
            x = (width * i) // 4
            y = height - random.randint(height//4, height//2)
            peak_points.append((x, y))
        
        peak_points.append((width, height))
        peak_points.insert(0, (0, height))
        
        if len(peak_points) >= 3:
            draw.polygon(peak_points, fill=colors['accent'])
        
        return image
    
    def _create_comedy_background(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Create a bright, cheerful comedy background"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Bright gradient
        image = self._create_gradient_background(image, colors)
        
        # Add some fun elements (circles and shapes)
        for _ in range(15):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(10, 40)
            
            # Semi-transparent circles
            circle_color = random.choice([colors['accent'], colors['secondary']])
            # Make it semi-transparent by blending
            circle_alpha = random.uniform(0.3, 0.7)
            
            draw.ellipse(
                [(x-radius, y-radius), (x+radius, y+radius)], 
                fill=circle_color
            )
        
        return image
    
    def _add_design_elements(self, image: Image.Image, colors: Dict[str, tuple], 
                           template: Dict[str, Any], genre: str) -> Image.Image:
        """Add design elements based on template"""
        
        elements = template.get('design_elements', [])
        
        for element in elements:
            if element == 'geometric_shapes':
                image = self._add_geometric_shapes(image, colors)
            elif element == 'frame_border':
                image = self._add_frame_border(image, colors)
            elif element == 'abstract_shapes':
                image = self._add_abstract_shapes(image, colors)
            elif element == 'decorative_elements':
                image = self._add_decorative_elements(image, colors, genre)
        
        return image
    
    def _add_geometric_shapes(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Add geometric shapes for modern style"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Add some triangles and rectangles
        for _ in range(3):
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
            
            shape_type = random.choice(['rectangle', 'triangle'])
            
            if shape_type == 'rectangle':
                draw.rectangle([(x1, y1), (x2, y2)], outline=colors['accent'], width=3)
            else:
                x3, y3 = x1 + random.randint(-50, 50), y2
                draw.polygon([(x1, y1), (x2, y1), (x3, y3)], outline=colors['accent'], width=2)
        
        return image
    
    def _add_frame_border(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Add a decorative frame border"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        border_width = 15
        
        # Outer border
        draw.rectangle(
            [(0, 0), (width-1, height-1)], 
            outline=colors['primary'], 
            width=border_width
        )
        
        # Inner border
        draw.rectangle(
            [(border_width//2, border_width//2), (width-border_width//2, height-border_width//2)], 
            outline=colors['accent'], 
            width=3
        )
        
        return image
    
    def _add_abstract_shapes(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Add abstract artistic shapes"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # Add flowing curves and organic shapes
        for _ in range(5):
            points = []
            center_x = random.randint(width//4, 3*width//4)
            center_y = random.randint(height//4, 3*height//4)
            
            for angle in range(0, 360, 30):
                radius = random.randint(30, 80)
                x = center_x + radius * math.cos(math.radians(angle))
                y = center_y + radius * math.sin(math.radians(angle))
                points.append((x, y))
            
            if len(points) >= 3:
                draw.polygon(points, outline=colors['accent'], width=2)
        
        return image
    
    def _add_decorative_elements(self, image: Image.Image, colors: Dict[str, tuple], genre: str) -> Image.Image:
        """Add genre-specific decorative elements"""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        if genre == 'Romance':
            # Add heart shapes
            for _ in range(3):
                x, y = random.randint(50, width-50), random.randint(50, height-50)
                self._draw_heart(draw, x, y, 20, colors['accent'])
        
        elif genre == 'Horror':
            # Add scary elements (jagged lines)
            for _ in range(8):
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = x1 + random.randint(-100, 100)
                y2 = y1 + random.randint(-100, 100)
                
                draw.line([(x1, y1), (x2, y2)], fill=colors['accent'], width=2)
        
        elif genre == 'Adventure':
            # Add compass-like elements
            for _ in range(2):
                x, y = random.randint(100, width-100), random.randint(100, height-100)
                self._draw_compass(draw, x, y, 30, colors['accent'])
        
        return image
    
    def _draw_heart(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: tuple):
        """Draw a heart shape"""
        # Simple heart using circles and triangle
        draw.ellipse([(x-size//2, y-size//2), (x, y)], fill=color)
        draw.ellipse([(x, y-size//2), (x+size//2, y)], fill=color)
        draw.polygon([(x-size//2, y), (x+size//2, y), (x, y+size)], fill=color)
    
    def _draw_compass(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: tuple):
        """Draw a compass rose"""
        # Draw compass points
        for angle in range(0, 360, 45):
            end_x = x + size * math.cos(math.radians(angle))
            end_y = y + size * math.sin(math.radians(angle))
            draw.line([(x, y), (end_x, end_y)], fill=color, width=2)
        
        # Draw center circle
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=color)
    
    def _add_text_elements(self, image: Image.Image, title: str, tagline: str, 
                          colors: Dict[str, tuple], template: Dict[str, Any]) -> Image.Image:
        """Add title and tagline text"""
        
        width, height = image.size
        
        # Calculate font sizes
        title_size = int(width * template['title_size_ratio'])
        tagline_size = int(width * template['tagline_size_ratio'])
        
        # Load fonts
        title_font = self._get_font(title_size)
        tagline_font = self._get_font(tagline_size)
        
        # Calculate positions
        title_x = int(width * template['title_position'][0])
        title_y = int(height * template['title_position'][1])
        tagline_x = int(width * template['tagline_position'][0])
        tagline_y = int(height * template['tagline_position'][1])
        
        # Create text overlay
        text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        
        # Add title with outline for better visibility
        title_color = colors['text'] + (255,)  # Add alpha
        outline_color = (255, 255, 255, 200) if sum(colors['text']) < 400 else (0, 0, 0, 200)
        
        self._draw_text_with_outline(text_draw, title, (title_x, title_y), title_font, 
                                   title_color, outline_color, center=True)
        
        # Add tagline
        tagline_color = colors['text'] + (200,)  # Slightly transparent
        self._draw_text_with_outline(text_draw, tagline, (tagline_x, tagline_y), tagline_font,
                                   tagline_color, outline_color, center=True)
        
        # Composite text layer onto image
        image = Image.alpha_composite(image.convert('RGBA'), text_layer)
        
        return image.convert('RGB')
    
    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get font with fallbacks"""
        for font_name in self.font_fallbacks:
            try:
                return ImageFont.truetype(f"{font_name}.ttf", size)
            except (OSError, IOError):
                continue
        
        # Ultimate fallback
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()
    
    def _draw_text_with_outline(self, draw: ImageDraw.Draw, text: str, position: tuple, 
                              font: ImageFont.ImageFont, fill_color: tuple, 
                              outline_color: tuple, center: bool = False):
        """Draw text with outline for better visibility"""
        x, y = position
        
        # Get text size for centering
        if center:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x -= text_width // 2
            y -= text_height // 2
        
        # Draw outline by drawing text in multiple positions
        outline_positions = [
            (x-2, y-2), (x-2, y), (x-2, y+2),
            (x, y-2), (x, y+2),
            (x+2, y-2), (x+2, y), (x+2, y+2)
        ]
        
        for outline_pos in outline_positions:
            draw.text(outline_pos, text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=fill_color)
    
    def _add_finishing_touches(self, image: Image.Image, genre: str, colors: Dict[str, tuple]) -> Image.Image:
        """Add final effects and touches"""
        
        # Apply genre-specific effects
        if genre == 'Horror':
            # Add slight blur and darkness
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
        
        elif genre == 'Romance':
            # Add soft glow
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.2)
        
        elif genre == 'Sci-Fi':
            # Add slight sharpening
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=20))
        
        # Add subtle vignette effect
        image = self._add_vignette(image, colors)
        
        return image
    
    def _add_vignette(self, image: Image.Image, colors: Dict[str, tuple]) -> Image.Image:
        """Add subtle vignette effect"""
        width, height = image.size
        
        # Create vignette mask
        vignette = Image.new('L', (width, height), 255)
        vignette_draw = ImageDraw.Draw(vignette)
        
        # Create radial gradient for vignette
        center_x, center_y = width // 2, height // 2
        max_distance = math.sqrt(center_x**2 + center_y**2) * 1.2
        
        for x in range(0, width, 4):
            for y in range(0, height, 4):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                opacity = max(0, min(255, int(255 * (1 - distance / max_distance))))
                
                vignette_draw.rectangle([(x, y), (x+3, y+3)], fill=opacity)
        
        # Apply vignette
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Convert to RGB and blend
        vignette_rgb = Image.merge('RGB', (vignette, vignette, vignette))
        image = Image.blend(image, vignette_rgb, 0.1)
        
        return image
    
    def _generate_tagline(self, summary: str, genre: str) -> str:
        """Generate a tagline based on summary and genre"""
        
        genre_taglines = {
            'Comedy': [
                "Laughter is the best medicine",
                "Life's too short to take seriously",
                "Sometimes the best moments are unplanned"
            ],
            'Drama': [
                "Every ending is a new beginning", 
                "The heart knows what matters most",
                "Some journeys change us forever"
            ],
            'Thriller': [
                "The truth has a way of surfacing",
                "Trust no one",
                "Some secrets refuse to stay buried"
            ],
            'Romance': [
                "Love finds a way",
                "Two hearts, one destiny", 
                "Sometimes love is worth the risk"
            ],
            'Horror': [
                "Fear has a new address",
                "Some doors should never be opened",
                "The nightmare is just beginning"
            ],
            'Adventure': [
                "Every legend starts with a dream",
                "Fortune favors the bold",
                "The greatest adventures await"
            ],
            'Sci-Fi': [
                "The future is what we make it",
                "Tomorrow brings new possibilities",
                "Progress comes with a price"
            ]
        }
        
        taglines = genre_taglines.get(genre, genre_taglines['Drama'])
        return random.choice(taglines)
    
    def _create_fallback_poster(self, title: str, genre: str) -> Dict[str, Any]:
        """Create a simple fallback poster"""
        try:
            width, height = 400, 600
            colors = self.color_schemes.get(genre, self.color_schemes['Drama'])
            
            # Create simple colored background
            image = Image.new('RGB', (width, height), colors['background'])
            draw = ImageDraw.Draw(image)
            
            # Add simple gradient
            for y in range(height):
                ratio = y / height
                r = int(colors['primary'][0] * (1-ratio) + colors['secondary'][0] * ratio)
                g = int(colors['primary'][1] * (1-ratio) + colors['secondary'][1] * ratio)
                b = int(colors['primary'][2] * (1-ratio) + colors['secondary'][2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Add title
            font = self._get_font(48)
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
            
            title_x = (width - title_width) // 2
            title_y = height // 2 - title_height // 2
            
            # Draw title with outline
            outline_color = (255, 255, 255) if sum(colors['text']) < 400 else (0, 0, 0)
            
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((title_x + dx, title_y + dy), title, font=font, fill=outline_color)
            
            draw.text((title_x, title_y), title, font=font, fill=colors['text'])
            
            # Add genre label
            genre_font = self._get_font(24)
            genre_text = f"A {genre} Film"
            genre_bbox = draw.textbbox((0, 0), genre_text, font=genre_font)
            genre_width = genre_bbox[2] - genre_bbox[0]
            
            genre_x = (width - genre_width) // 2
            genre_y = title_y + title_height + 40
            
            draw.text((genre_x, genre_y), genre_text, font=genre_font, fill=colors['text'])
            
            # Convert to base64
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            data_url = f"data:image/png;base64,{img_base64}"
            
            return {
                'image': data_url,
                'image_bytes': img_bytes.getvalue(),
                'description': f"Simple {genre.lower()} movie poster",
                'style': 'fallback',
                'genre': genre
            }
            
        except Exception as e:
            st.error(f"Fallback poster creation failed: {e}")
            return None