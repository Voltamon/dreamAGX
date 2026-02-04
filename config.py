"""
Configuration settings for the hologram display application.
"""

# Display settings
DEFAULT_SIZE = 1200  # Default square size in pixels (increased for better visibility)
MIN_SIZE = 400       # Minimum square size
MAX_SIZE = 1800      # Maximum square size
SCALE_STEP = 50      # Pixels to scale per keypress

# Performance settings
FPS = 60            # Frames per second

# Colors (R, G, B)
BACKGROUND_COLOR = (0, 0, 0)      # Black background
BORDER_COLOR = (50, 50, 50)       # Dark gray borders
DEMO_COLOR_1 = (255, 100, 150)    # Pink
DEMO_COLOR_2 = (100, 200, 255)    # Light blue
DEMO_COLOR_3 = (150, 255, 100)    # Light green

# Demo animation settings
ROTATION_SPEED = 2.0   # Degrees per frame
PULSE_SPEED = 0.05     # Pulse animation speed

# Screen capture settings
CAPTURE_MODE = "fullscreen"  # fullscreen, region, or window
CAPTURE_FPS = 60             # How often to capture screen (matches display FPS)

# Tkinter-compatible colors (hex format)
BG_COLOR = "#000000"
BORDER_COLOR = "#323232"
FRAME_DELAY_MS = 1000 // FPS  # Convert FPS to milliseconds

# Display layout
CENTER_GAP = 40  # Size of the center square gap in pixels
