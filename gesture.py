import cv2
import mediapipe as mp
import numpy as np
import os
import mediapipe
print(f"Loading MediaPipe from: {mediapipe.__file__}")
# --- 1. SETUP & CONFIGURATION ---
mp_hands = mp.solutions.hands
# Max hands = 2 (Left for Mode, Right for Action)
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Output file for OpenSCAD
SCAD_FILE = "design_output.scad"

# State Variables
trajectory = []  # Stores the path of the right index finger
is_drawing = False

# --- 2. HELPER FUNCTIONS ---

def is_left_hand_open(hand_lms):
    """
    Detects if the Left Hand is in 'Type Posture' (Open Palm).
    Returns True if fingers are up, False otherwise.
    """
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
    # Check if tips are above MCP joints (Y coordinate is smaller at top of screen)
    for tip in tips:
        if hand_lms.landmark[tip].y > hand_lms.landmark[tip - 2].y:
            return False
    return True

def recognize_shape(points):
    """
    Simple heuristic to recognize shape from trajectory points.
    In a full implementation, this would be an HMM or Neural Network.
    """
    if len(points) < 10: return None
    
    # Calculate bounding box
    pts = np.array(points)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Simple Logic: 
    # If width and height are similar -> Square/Circle
    # If very wide or tall -> Rectangle
    ratio = width / (height + 1e-5)
    
    if 0.8 < ratio < 1.2:
        return "cube([20, 20, 20], center=true);" # Return OpenSCAD command
    else:
        return "cylinder(h=20, r=10, center=true);"

def update_scad_file(command):
    """Writes the command to the .scad file for Real-time Preview"""
    with open(SCAD_FILE, "w") as f:
        f.write(f"{command}")
    print(f"Updated SCAD with: {command}")

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    left_hand_active = False
    
    if results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Identify Hand (Left or Right)
            # Note: MediaPipe assumes mirrored image, so 'Left' label might appear for your actual right hand depending on flip.
            # Adjust 'label' check based on your camera view.
            label = results.multi_handedness[idx].classification[0].label
            
            # Draw Skeleton
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # --- LOGIC A: LEFT HAND (Type Posture) ---
            # If Left Hand is present and Open, we enter "DESIGN MODE"
            if label == 'Left':
                if is_left_hand_open(hand_lms):
                    left_hand_active = True
                    cv2.putText(frame, "MODE: DRAW", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # --- LOGIC B: RIGHT HAND (Action Gesture) ---
            # If Right Hand is present AND Left Hand is triggering mode
            if label == 'Right' and left_hand_active:
                is_drawing = True
                # Track Index Finger Tip (Landmark 8)
                x = int(hand_lms.landmark[8].x * w)
                y = int(hand_lms.landmark[8].y * h)
                trajectory.append([x, y])
                
                # Visualize Drawing on Screen
                pts = np.array(trajectory, np.int32)
                cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

    # --- 4. EXECUTION (When Gesture Ends) ---
    # If Left Hand drops (trigger released) and we have data, generate the shape
    if not left_hand_active and is_drawing:
        scad_cmd = recognize_shape(trajectory)
        if scad_cmd:
            update_scad_file(scad_cmd)
        
        # Reset Logic
        trajectory = []
        is_drawing = False

    cv2.imshow("HG3D Prototype", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
