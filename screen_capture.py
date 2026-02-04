"""
Screen capture module for hologram display.
Provides fast full-screen capture using mss library.
"""

import mss
from PIL import Image, ImageTk
import tkinter as tk


class ScreenCapture:
    """Handles screen capture operations for hologram display."""
    
    def __init__(self):
        """Initialize the screen capture system."""
        self.sct = mss.mss()
        
        # Monitor selection for capture
        # 0 = all monitors, 1 = primary (left/main), 2 = secondary (right)
        try:
            capture_monitor = self.sct.monitors[2]  # Secondary monitor
            print(f"✓ Using secondary monitor for capture")
            print(f"  Monitor bounds: {capture_monitor['width']}x{capture_monitor['height']}")
            
            # ===== CAPTURE REGION CONFIGURATION =====
            # This defines what area to capture FROM THE SECONDARY MONITOR
            # The hologram display will be on PRIMARY monitor, so no overlap!
            
            # Option 1: Capture entire secondary monitor (uncomment to use)
            # self.custom_region = None
            
            # Option 2: Capture specific area from secondary monitor (currently active)
            self.custom_region = {
                'left': capture_monitor['left'] + 200,    # Offset from left edge of secondary monitor
                'top': capture_monitor['top'] + 100,      # Offset from top edge of secondary monitor  
                'width': 1200,                             # Width of capture area
                'height': 900                              # Height of capture area
            }
            
            self.monitor = capture_monitor
            
        except IndexError:
            # Fallback to primary if secondary doesn't exist
            self.monitor = self.sct.monitors[1]
            print("⚠ Secondary monitor not found, using primary monitor")
            print("  WARNING: This may cause recursive display!")
            self.custom_region = None
        
    def set_region(self, x, y, width, height):
        """
        Set a custom region to capture.
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
        """
        self.custom_region = {
            'left': int(x),
            'top': int(y),
            'width': int(width),
            'height': int(height)
        }
    
    def reset_region(self):
        """Reset to full screen capture."""
        self.custom_region = None
        
    def get_full_screen(self):
        """
        Capture the screen (full or custom region) and return as PIL Image.
        
        Returns:
            PIL.Image: Captured screen image
        """
        # Use custom region if set, otherwise full screen
        monitor = self.custom_region if self.custom_region else self.monitor
        
        # Capture the screen
        screenshot = self.sct.grab(monitor)
        
        # Convert to PIL Image
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        
        return img
    
    def get_screen_for_quadrant(self, target_size):
        """
        Capture screen and resize for hologram quadrant display.
        Fills the entire square by scaling and cropping - no black bars.
        
        Args:
            target_size: Size of each quadrant (width, height tuple)
            
        Returns:
            PIL.Image: Resized screen capture that fills the quadrant
        """
        img = self.get_full_screen()
        
        # Get screen dimensions
        width, height = img.size
        target_w, target_h = target_size
        
        # Scale to FILL the entire square (may crop edges)
        # This eliminates black bars by using the larger scale factor
        scale = max(target_w / width, target_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the full screen
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to exact square size (centered crop)
        left = (new_width - target_w) // 1.5
        top = (new_height - target_h) // 1.5
        right = (left + target_w)//1.5
        bottom = (top + target_h)//1.5
        
        img = img.crop((left, top, right, bottom))
        
        return img
    
    def cleanup(self):
        """Clean up resources."""
        self.sct.close()
