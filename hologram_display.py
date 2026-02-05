"""
Hologram Display Application - Screen Capture Version
Displays live screen capture in 4 directions for hologram pyramid projection.
Streams fullscreen content at 60 FPS.
"""

import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import sys
from config import *
from screen_capture import ScreenCapture
from region_selector import RegionSelector


class HologramDisplay:
    def __init__(self, root):
        self.root = root
        self.root.title("Hologram Display - Screen Capture")
        self.root.configure(bg=BG_COLOR)
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Position window on PRIMARY monitor to avoid capturing itself
        # This prevents the infinite mirror effect
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        # Make window fullscreen AFTER positioning
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.quit_app())
        self.root.bind('q', lambda e: self.quit_app())
        
        # Initial square size
        self.square_size = DEFAULT_SIZE
        
        # Create canvas
        self.canvas = Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Initialize screen capture
        try:
            self.screen_capture = ScreenCapture()
            self.capture_enabled = True
            print("✓ Screen capture initialized")
        except Exception as e:
            print(f"✗ Screen capture failed: {e}")
            print("  Please install dependencies: pip install mss pillow")
            self.capture_enabled = False
            self.quit_app()
            return
        
        # Bind keyboard events
        self.root.bind('<plus>', lambda e: self.scale_up())
        self.root.bind('<equal>', lambda e: self.scale_up())
        self.root.bind('<minus>', lambda e: self.scale_down())
        self.root.bind('<Up>', lambda e: self.scale_up())
        self.root.bind('<Down>', lambda e: self.scale_down())
        self.root.bind('h', lambda e: self.toggle_instructions())
        self.root.bind('r', lambda e: self.select_region())
        self.root.bind('f', lambda e: self.reset_to_fullscreen())
        
        # Instructions visibility
        self.show_instructions = True
        
        # Cache for PhotoImage objects (prevent garbage collection)
        self.photo_cache = []
        
        # Frame counter for FPS display
        self.frame_count = 0
        self.fps_display = FPS
        
        # Start animation loop
        self.animate()
    
    def get_square_offset(self):
        """Calculate offset to center the square display."""
        offset_x = (self.screen_width - self.square_size) // 2
        offset_y = (self.screen_height - self.square_size) // 2
        return offset_x, offset_y
    
    def scale_up(self):
        """Increase the display size."""
        max_size = min(MAX_SIZE, self.screen_width, self.screen_height)
        self.square_size = min(self.square_size + SCALE_STEP, max_size)
    
    def scale_down(self):
        """Decrease the display size."""
        self.square_size = max(self.square_size - SCALE_STEP, MIN_SIZE)
    
    def toggle_instructions(self):
        """Toggle instruction visibility."""
        self.show_instructions = not self.show_instructions
    
    def select_region(self):
        """Open region selector to choose capture area."""
        print("\nOpening region selector...")
        print("DRAG to select area, ENTER to confirm, ESC to cancel")
        
        selector = RegionSelector()
        region = selector.select_region()
        
        if region:
            x, y, width, height = region
            self.screen_capture.set_region(x, y, width, height)
            print(f"✓ Region set: {width}x{height} at ({x}, {y})")
            print("  Press 'F' to return to full screen")
        else:
            print("✗ Region selection cancelled")
    
    def reset_to_fullscreen(self):
        """Reset to full screen capture."""
        self.screen_capture.reset_region()
        print("✓ Reset to full screen capture")
    
    def quit_app(self):
        """Exit the application."""
        if self.capture_enabled:
            self.screen_capture.cleanup()
        self.root.quit()
    
    def create_rotated_image(self, img, rotation_deg):
        """
        Rotate an image by specified degrees.
        
        Args:
            img: PIL Image
            rotation_deg: Rotation angle in degrees
            
        Returns:
            PIL Image: Rotated image
        """
        if rotation_deg == 0:
            return img
        return img.rotate(-rotation_deg, expand=False)
    
    def draw_hologram_display(self):
        """Draw the 4-directional hologram display with live screen capture."""
        offset_x, offset_y = self.get_square_offset()
        half_size = self.square_size // 2
        gap = CENTER_GAP // 2  # Half gap on each side of center
        
        # Capture screen - quadrants are smaller to make room for gap
        quadrant_size = (half_size - gap, half_size - gap)
        captured_img = self.screen_capture.get_screen_for_quadrant(quadrant_size)
        
        # Clear photo cache from previous frame
        self.photo_cache.clear()
        
        # Calculate center point (intersection of the square)
        center_x = offset_x + half_size
        center_y = offset_y + half_size
        
        # Calculate quadrant dimensions
        quad_w = half_size - gap
        quad_h = half_size - gap
        
        # Create rotated versions for each quadrant
        # Position each quadrant from the outer edge toward the center
        rotations = [
            # Bottom: center horizontally, position from bottom edge
            (180, center_x, offset_y + half_size + gap + quad_h // 2+20, True),
            # Top: center horizontally, position from top edge  
            (0, center_x, offset_y + half_size - gap - quad_h // 2-20, False),
            # Left: center vertically, position from left edge
            (90, offset_x -20 + half_size - gap - quad_w // 2, center_y, True),
            # Right: center vertically, position from right edge
            (90, offset_x + 20 + half_size + gap + quad_w // 2, center_y, False)
        ]
        
        for rotation, x, y, flip in rotations:
            rotated = self.create_rotated_image(captured_img, rotation)
            if flip:
                rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)
            photo = ImageTk.PhotoImage(rotated)
            self.photo_cache.append(photo)  # Prevent garbage collection
            self.canvas.create_image(x, y, image=photo, anchor='center')
        
        # Draw center square gap
        self.canvas.create_rectangle(
            center_x - gap, center_y - gap,
            center_x + gap, center_y + gap,
            fill=BG_COLOR, outline=BORDER_COLOR, width=2
        )
        
        # Draw border around square (no center lines)
        self.canvas.create_rectangle(
            offset_x, offset_y,
            offset_x + self.square_size, offset_y + self.square_size,
            outline=BORDER_COLOR, width=2
        )
    
    def draw_instructions(self):
        """Draw on-screen instructions."""
        if self.show_instructions:
            mode = "Region" if self.screen_capture.custom_region else "Full Screen"
            instructions = [
                f"{mode} | Size: {self.square_size}px | {FPS} FPS",
                "Controls:",
                "+/- or ↑/↓: Scale | R: Select Region",
                "F: Full Screen | H: Hide | Q: Quit"
            ]
            
            y_offset = 10
            for text in instructions:
                self.canvas.create_text(
                    10, y_offset,
                    text=text,
                    fill="#00FF00",
                    font=("Arial", 11, "bold"),
                    anchor="nw"
                )
                y_offset += 20
    
    def animate(self):
        """Main animation loop - captures and displays screen at 60 FPS."""
        if not self.capture_enabled:
            return
        
        try:
            # Clear canvas
            self.canvas.delete("all")
            
            # Draw hologram display with screen capture
            self.draw_hologram_display()
            
            # Draw instructions
            self.draw_instructions()
            
            # Update frame counter
            self.frame_count += 1
            
        except Exception as e:
            print(f"Error in animation loop: {e}")
        
        # Schedule next frame
        self.root.after(FRAME_DELAY_MS, self.animate)


def main():
    print("="*60)
    print("Hologram Display - Full Screen Capture Mode")
    print("="*60)
    print(f"Running at {FPS} FPS")
    print("\nCapturing your entire screen in real-time!")
    print("Perfect for streaming Discord, games, videos, etc.\n")
    print("Controls:")
    print("  +/- or ↑/↓  : Scale display size")
    print("  H          : Hide/Show instructions")
    print("  ESC or Q   : Quit")
    print("="*60)
    print("\nStarting capture...")
    
    root = tk.Tk()
    app = HologramDisplay(root)
    root.mainloop()


if __name__ == "__main__":
    main()
