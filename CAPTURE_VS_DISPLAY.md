# Capture Region vs Display Separation

## How It Works Now:

### HOLOGRAM DISPLAY (Output)
- **Location**: PRIMARY monitor (left/main screen)
- **Purpose**: Shows the 4-directional hologram
- **Fullscreen**: Yes, on primary monitor only

### SCREEN CAPTURE (Input)  
- **Location**: SECONDARY monitor (right screen)
- **Purpose**: Captures content to display in hologram
- **Region**: Specific area on secondary monitor

## This Prevents Infinite Mirror!

```
┌─────────────────┐  ┌─────────────────┐
│ PRIMARY MONITOR │  │ SECONDARY       │
│                 │  │ MONITOR         │
│  ┌───────────┐  │  │  ┌──────────┐   │
│  │ HOLOGRAM  │  │  │  │ CAPTURE  │   │
│  │ DISPLAY   │  │  │  │ REGION   │   │
│  │ (Output)  │  │  │  │ (Input)  │   │
│  └───────────┘  │  │  └──────────┘   │
│                 │  │                 │
└─────────────────┘  └─────────────────┘
```

**No overlap = No infinite loop!**

## To Adjust Capture Area:

Edit `screen_capture.py` around line 28-32:

```python
self.custom_region = {
    'left': capture_monitor['left'] + 200,  # Offset from secondary monitor left
    'top': capture_monitor['top'] + 100,    # Offset from secondary monitor top
    'width': 1200,  # Width
    'height': 900   # Height
}
```

**Note**: The `left` and `top` values are absolute screen coordinates on the secondary monitor.

## Single Monitor Users:

If you only have ONE monitor:
1. Press **R** key while app is running
2. Select ONLY your content area (not the hologram window)
3. Press ENTER

This ensures capture region ≠ display window!
