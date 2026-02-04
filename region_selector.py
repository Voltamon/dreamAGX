"""
Region Selector - Interactive screen region selection tool
"""

import tkinter as tk
from tkinter import messagebox


class RegionSelector:
    """Interactive region selection tool."""
    
    def __init__(self):
        self.region = None
        
    def select_region(self):
        """
        Open a fullscreen overlay to select a region.
        Returns (x, y, width, height) or None if cancelled.
        """
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)
        root.configure(bg='black')
        root.attributes('-topmost', True)
        
        canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        
        # Instructions
        instructions = canvas.create_text(
            root.winfo_screenwidth() // 2, 50,
            text="DRAG to select region | ESC to cancel | ENTER when done",
            fill='white',
            font=('Arial', 16, 'bold')
        )
        
        # Selection state
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection_complete = False
        
        def on_mouse_down(event):
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                canvas.delete(self.rect)
        
        def on_mouse_drag(event):
            if self.start_x is not None:
                if self.rect:
                    canvas.delete(self.rect)
                self.rect = canvas.create_rectangle(
                    self.start_x, self.start_y,
                    event.x, event.y,
                    outline='red',
                    width=3
                )
                # Show dimensions
                w = abs(event.x - self.start_x)
                h = abs(event.y - self.start_y)
                canvas.delete('dim_text')
                canvas.create_text(
                    (self.start_x + event.x) // 2,
                    (self.start_y + event.y) // 2,
                    text=f"{w} x {h}",
                    fill='yellow',
                    font=('Arial', 14, 'bold'),
                    tags='dim_text'
                )
        
        def on_mouse_up(event):
            if self.start_x is not None:
                # Calculate region
                x1 = min(self.start_x, event.x)
                y1 = min(self.start_y, event.y)
                x2 = max(self.start_x, event.x)
                y2 = max(self.start_y, event.y)
                
                width = x2 - x1
                height = y2 - y1
                
                if width > 50 and height > 50:  # Minimum size
                    self.region = (x1, y1, width, height)
                    self.selection_complete = True
        
        def on_key(event):
            if event.keysym == 'Escape':
                root.quit()
            elif event.keysym == 'Return' and self.selection_complete:
                root.quit()
        
        canvas.bind('<ButtonPress-1>', on_mouse_down)
        canvas.bind('<B1-Motion>', on_mouse_drag)
        canvas.bind('<ButtonRelease-1>', on_mouse_up)
        root.bind('<Key>', on_key)
        
        root.mainloop()
        root.destroy()
        
        return self.region
