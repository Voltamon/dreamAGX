import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import open3d as o3d
import sys
import ctypes

# ==============================================================================
# Global State
# ==============================================================================
class GestureState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        
        # Transformation Control
        self.rotation_x = 0.0 # Accumulated rotation around global X
        self.rotation_y = 0.0 # Accumulated rotation around global Y
        self.scale = 1.0      # Current scale factor (starts at 1.0)
        self.trans_x = 0.0    # Accumulated translation X
        self.trans_y = 0.0    # Accumulated translation Y
        self.target_look_at = np.zeros(3) # Focus point
        
        # Frame-to-frame deltas (or absolute tracking ref)
        self.last_zoom_dist = None
        self.last_pan_pos = None # (x, y) midpoint for panning
        self.last_rotate_pos = None # (x, y)
        
        # Cursor Mode
        self.cursor_active = False
        self.cursor_pos = (0, 0) # Normalized (0.0 - 1.0)
        
        # Menu
        self.menu_active = False
        
        # Display Info
        self.status_text = "Idle"
        self.status_sub_text = ""
        self.active_gestures = [] # List of strings for debug

state = GestureState()

# Get Screen Resolution
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)
HALF_W = SCREEN_W // 2

# ==============================================================================
# Gesture Definitions
# ==============================================================================
# Finger indices
# THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4
# INDEX_MCP = 5, ... TIP = 8
# MIDDLE_MCP = 9, ... TIP = 12
# RING_MCP = 13, ... TIP = 16
# PINKY_MCP = 17, ... TIP = 20

def is_finger_curled(landmarks, tip_idx, mcp_idx):
    # Simple check: if tip is lower (higher y value) than MCP, it's curled (assuming palm facing camera)
    # Better check: distance to wrist vs MCP distance to wrist
    wrist = landmarks[0]
    tip = landmarks[tip_idx]
    mcp = landmarks[mcp_idx]
    
    # Distance method
    d_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
    d_mcp = (mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2
    return d_tip < d_mcp

def detect_hand_pose(landmarks):
    """
    Returns: 'PINCH', 'GRAB', 'ONE', 'PEACE', 'OPEN', 'UNKNOWN'
    """
    # Check fingers status
    # Thumb is special, we check if it's close to Index Tip
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Distances for Pinch
    pinch_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
    # Curl checks (Index, Middle, Ring, Pinky)
    index_curled = is_finger_curled(landmarks, 8, 5)
    middle_curled = is_finger_curled(landmarks, 12, 9)
    ring_curled = is_finger_curled(landmarks, 16, 13)
    pinky_curled = is_finger_curled(landmarks, 20, 17)
    
    # Count curled fingers (excluding thumb)
    curled_count = sum([index_curled, middle_curled, ring_curled, pinky_curled])
    
    # 1. Grab [TIMRP] - Leniency: 3 or more fingers curled
    if curled_count >= 3:
        # Also check thumb? If thumb is far out it might be "Thumbs Up".
        # But for now, let's assume if 3 fingers are curled it's likely a grab intent vs Pinch.
        # But Pinch involves Index+Thumb. If Index is curled, it's not a Pinch.
        if index_curled and middle_curled:
             return 'GRAB'
        # If Index is not curled, it might be pointing.
        if curled_count == 4:
            return 'GRAB'
    
    # 2. Pinch [TM]
    if pinch_dist < 0.05:
        return 'PINCH'
        
    # 3. One [TMRP] - Index extended, others curled
    if not index_curled and middle_curled and ring_curled and pinky_curled:
        return 'ONE'
        
    # 4. Peace [TRP] - Index + Middle extended, others curled
    # User Request: "Index and Middle close, others close, two clusters far apart"
    # Logic: Index & Middle Extended. Ring & Pinky "Down" (Curled OR significantly shorter than Index/Middle)
    if not index_curled and not middle_curled:
        # Strict curl check failed for some users?
        # Let's check relative extension.
        # Wrist Distances
        d_idx = (index_tip.x - landmarks[0].x)**2 + (index_tip.y - landmarks[0].y)**2
        d_mid = (middle_tip.x - landmarks[0].x)**2 + (middle_tip.y - landmarks[0].y)**2
        d_ring = (ring_tip.x - landmarks[0].x)**2 + (ring_tip.y - landmarks[0].y)**2
        d_pinky = (pinky_tip.x - landmarks[0].x)**2 + (pinky_tip.y - landmarks[0].y)**2
        
        avg_active = (d_idx + d_mid) / 2
        avg_inactive = (d_ring + d_pinky) / 2
        
        # If active fingers are significantly more extended than inactive ones
        if avg_active > 1.5 * avg_inactive:
             return 'PEACE'
             
        # Fallback to strict curled check if ratio method is ambiguous but they are curled
        if ring_curled and pinky_curled:
            return 'PEACE'
    
    return 'OPEN'

# ==============================================================================
# Hand Tracking (Persistent IDs)
# ==============================================================================
class HandTracker:
    def __init__(self):
        self.hands = {} # id -> {'landmark': obj, 'centroid': (x, y), 'lost_frames': 0}
        self.next_id = 0
        self.max_lost_frames = 5
        self.role_map = {} # 'primary': id, 'secondary': id

    def update(self, multi_hand_landmarks):
        current_centroids = []
        if multi_hand_landmarks:
            for hand in multi_hand_landmarks:
                # Calculate centroid (avg of all points or just wrist? wrist is stable)
                # Wrist is 0. Let's use wrist for simple stable tracking.
                wrist = hand.landmark[0]
                current_centroids.append({'pos': (wrist.x, wrist.y), 'data': hand})

        # Match with existing hands
        active_ids = []
        
        # Simple greedy matching
        # For each new detection, find closest existing hand
        # If dist < threshold, match.
        
        assigned_new_indices = set()
        
        for hid, hdata in list(self.hands.items()):
            best_dist = 0.2 # Threshold
            best_idx = -1
            
            hx, hy = hdata['centroid']
            
            for idx, item in enumerate(current_centroids):
                if idx in assigned_new_indices: continue
                
                cx, cy = item['pos']
                dist = np.sqrt((hx - cx)**2 + (hy - cy)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx != -1:
                # Matched
                self.hands[hid]['landmark'] = current_centroids[best_idx]['data']
                self.hands[hid]['centroid'] = current_centroids[best_idx]['pos']
                self.hands[hid]['lost_frames'] = 0
                assigned_new_indices.add(best_idx)
                active_ids.append(hid)
            else:
                # Lost
                self.hands[hid]['lost_frames'] += 1
                if self.hands[hid]['lost_frames'] > self.max_lost_frames:
                    del self.hands[hid]

        # Register new hands
        for idx, item in enumerate(current_centroids):
            if idx not in assigned_new_indices:
                hid = self.next_id
                self.next_id += 1
                self.hands[hid] = {
                    'landmark': item['data'],
                    'centroid': item['pos'],
                    'lost_frames': 0
                }
                active_ids.append(hid)
        
        # Assign Roles
        # If we have 1 active hand -> Primary
        # If we have 2 active hands -> Oldest is Primary? Or first assigned?
        # Requirement: "If there is only one hand, it should taken as primary. If a second comes, secondary."
        
        # Let's sort active_ids by their numeric ID (creation time)
        active_ids.sort()
        
        primary_hand = None
        secondary_hand = None
        
        if len(active_ids) > 0:
            primary_id = active_ids[0]
            primary_hand = self.hands[primary_id]['landmark']
            
            if len(active_ids) > 1:
                secondary_id = active_ids[1]
                secondary_hand = self.hands[secondary_id]['landmark']
        
        return primary_hand, secondary_hand


# ==============================================================================
# CV Thread (MediaPipe)
# ==============================================================================
def cv_thread_func():
    global state
    
    mp_hands = mp.solutions.hands
    
    # Try multiple camera indices
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera 1 failed, trying Camera 0...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        with state.lock:
            state.running = False
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.6
    )

    print("CV Thread Started. Press ESC in the CV window to exit.")
    
    # Setup Window
    window_name = 'Hand Sense'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, HALF_W, SCREEN_H)
    cv2.moveWindow(window_name, 0, 0)
    
    # Smoothing State
    # We smooth the *deltas* slightly to reduce jitter
    def apply_deadzone(value, threshold):
        return 0.0 if abs(value) < threshold else value

    # Tracker instance
    tracker = HandTracker()

    while True:
        with state.lock:
            if not state.running:
                break
                
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        # Flip and Convert
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape
        
        results = hands.process(rgb_frame)
        
        # Defaults
        status = "Idle"
        sub_status = ""
        active_gestures_list = []
        
        primary_hand, secondary_hand = tracker.update(results.multi_hand_landmarks)
        
        if primary_hand or secondary_hand:
            # We have at least one hand
            
            # --- Draw Landmarks (All detected) ---
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Detect Poses ---
            pose_p = 'NONE'
            pose_s = 'NONE'
            
            if primary_hand:
                pose_p = detect_hand_pose(primary_hand.landmark)
            if secondary_hand:
                pose_s = detect_hand_pose(secondary_hand.landmark)
                
            active_gestures_list.append(f"P: {pose_p}")
            active_gestures_list.append(f"S: {pose_s}")
            
            # --- Mapping Logic ---
            
            with state.lock:
                # F. Menu [Clap -> P5 + S5]
                # Detecting Clap usually involves distance between wrists/palms
                is_clapping = False
                if primary_hand and secondary_hand:
                    p_wrist = primary_hand.landmark[0]
                    s_wrist = secondary_hand.landmark[0]
                    wrist_dist = np.sqrt((p_wrist.x - s_wrist.x)**2 + (p_wrist.y - s_wrist.y)**2)
                    
                    # Increased threshold for Clap (was 0.15)
                    if wrist_dist < 0.25: 
                        is_clapping = True
                        status = "MENU (Clap)"
                        state.menu_active = True
                        
                        # Trigger once logic? Or continuous resets?
                        # Continuous reset feels responsive for "Start Over"
                        state.trans_x = 0
                        state.trans_y = 0
                        state.rotation_x = 0
                        state.rotation_y = 0
                        state.scale = 1.0
                
                if not is_clapping:
                    DEADZONE = 0.002 # 0.2% of screen
                    
                    # D. Zoom [P1 + S1] -> Pinch on both hands
                    if pose_p == 'PINCH' and pose_s == 'PINCH':
                        status = "ZOOM"
                        p1 = primary_hand.landmark[4] # Thumb P
                        p2 = secondary_hand.landmark[4] # Thumb S
                        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                        
                        if state.last_zoom_dist is not None:
                            ratio = dist / (state.last_zoom_dist + 1e-6)
                            # Apply 3x Sensitivity to Zoom Speed
                            # NewScale = OldScale * (1 + (ratio-1)*3)
                            
                            delta = ratio - 1.0
                            # Deadzone for Zoom
                            if abs(delta) < 0.005: delta = 0
                            
                            active_ratio = 1.0 + (delta * 3.0)
                            
                            # Dampening/Limits
                            if 0.5 < active_ratio < 1.5:
                                state.scale *= active_ratio
                            state.scale = max(0.1, min(state.scale, 5.0))
                            sub_status = f"{state.scale:.2f}x"
                        state.last_zoom_dist = dist
                    else:
                        state.last_zoom_dist = None
                    
                    # A. Focus [A - P1] -> Pinch on P only? 
                    # Actually prompt says A - P1. If Pinch is held, maybe we center/focus?
                    # Let's say checks for just P1 active (and S not interfering)
                    if pose_p == 'PINCH' and pose_s != 'PINCH':
                        # Continuous pinch might mean "Hold Focus" or just trigger once?
                        # Let's implement as "Reset View to Center" if held for a bit or just strict checking
                        # Or maybe it just locks the view?
                        # Interpretation: "Focus" -> Recenter Object
                        status = "FOCUS"
                        state.trans_x = 0.0
                        state.trans_y = 0.0
                        
                    # B. Rotate [B - P2] -> Grab on P
                    if pose_p == 'GRAB':
                        status = "ROTATE"
                        # Use wrist position for movement delta
                        curr_x = primary_hand.landmark[0].x
                        curr_y = primary_hand.landmark[0].y
                        
                        if state.last_rotate_pos is not None:
                            dx = curr_x - state.last_rotate_pos[0]
                            dy = curr_y - state.last_rotate_pos[1]
                            
                            dx = apply_deadzone(dx, DEADZONE)
                            dy = apply_deadzone(dy, DEADZONE)
                            
                            # 3x Sensitivity (Was 4.0 -> 9.0)
                            SENSITIVITY = 9.0
                            state.rotation_y += dx * SENSITIVITY # Move X -> Rot Y
                            state.rotation_x += dy * SENSITIVITY # Move Y -> Rot X
                        
                        state.last_rotate_pos = (curr_x, curr_y)
                    else:
                        state.last_rotate_pos = None

                    # C. Drag [C - P4] -> Peace on P
                    if pose_p == 'PEACE':
                        status = "DRAG"
                        # Use midpoint of index/middle or wrist
                        curr_x = primary_hand.landmark[0].x
                        curr_y = primary_hand.landmark[0].y
                        
                        if state.last_pan_pos is not None:
                            dx = curr_x - state.last_pan_pos[0]
                            dy = curr_y - state.last_pan_pos[1]
                            
                            dx = apply_deadzone(dx, DEADZONE)
                            dy = apply_deadzone(dy, DEADZONE)
                            
                            # 3x Sensitivity (Was 2.0 -> 6.0)
                            PAN_SENSITIVITY = 6.0
                            state.trans_x += dx * PAN_SENSITIVITY
                            state.trans_y -= dy * PAN_SENSITIVITY # Invert Y
                            
                            # BOUNDS CHECKING
                            # Arbitrary bounds [-1.5, 1.5] for X, [-1.0, 1.0] for Y (approx screen space)
                            BOUND_X = 1.5
                            BOUND_Y = 1.0
                            
                            state.trans_x = max(-BOUND_X, min(state.trans_x, BOUND_X))
                            state.trans_y = max(-BOUND_Y, min(state.trans_y, BOUND_Y))
                        
                        state.last_pan_pos = (curr_x, curr_y)
                    else:
                        state.last_pan_pos = None
                        
                    # E. Cursor [E - S3] -> One on S
                    if pose_s == 'ONE':
                        # Track Index Tip 
                        idx_tip = secondary_hand.landmark[8]
                        state.cursor_pos = (idx_tip.x, idx_tip.y)
                        state.cursor_active = True
                        status = "CURSOR" if status == "Idle" else status + " + CURSOR"
                        
                        # Draw visuals
                        cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
                        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    else:
                        state.cursor_active = False

                state.status_text = status
                state.status_sub_text = sub_status
                state.active_gestures = active_gestures_list

        else:
            with state.lock:
                state.active_gestures = []
                state.status_text = "Idle"
                state.last_zoom_dist = None
                state.last_rotate_pos = None
                state.last_pan_pos = None

        # Draw UI
        cv2.putText(frame, f"State: {state.status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if state.status_sub_text:
            cv2.putText(frame, state.status_sub_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
            
        y_off = 120
        for g in state.active_gestures:
            cv2.putText(frame, g, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            y_off += 30

        # Draw Legends
        lines = [
            "Right (P):",
            "- Pinch: Focus (Reset Pan)",
            "- Grab: Rotate",
            "- Peace: Drag (Pan)",
            "Left (S):",
            "- One: Cursor",
            "Both:",
            "- 2x Pinch: Zoom",
            "- Clap: Menu (Reset All)"
        ]
        
        start_y = h - 200
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, start_y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(5) & 0xFF == 27: # Esc
            with state.lock:
                state.running = False
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==============================================================================
# Main Thread (Open3D Visualizer)
# ==============================================================================
def main():
    # Start CV Thread
    t = threading.Thread(target=cv_thread_func)
    t.daemon = True
    t.start()
    
    # Open3D Setup
    print("Initializing Open3D...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Gesture Controlled CAD", width=HALF_W, height=SCREEN_H, left=HALF_W, top=0)
    
    # Create Geometry - Box
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.2, 0.2]) # Reddish
    mesh.translate((-0.5, -0.5, -0.5))
    
    # Coordinate Frame
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5)
    
    vis.add_geometry(mesh)
    vis.add_geometry(coords)
    
    # State tracking
    last_rot_x = 0.0
    last_rot_y = 0.0
    current_scale = 1.0
    last_trans_x = 0.0
    last_trans_y = 0.0
    
    print("Open3D Window logic starting...")
    while True:
        with state.lock:
            if not state.running:
                break
            
            target_rx = state.rotation_x
            target_ry = state.rotation_y
            target_scale = state.scale
            target_tx = state.trans_x
            target_ty = state.trans_y
            
        # 1. Rotation
        d_rx = target_rx - last_rot_x
        d_ry = target_ry - last_rot_y
        
        if abs(d_rx) > 1e-4 or abs(d_ry) > 1e-4:
            R = mesh.get_rotation_matrix_from_xyz((d_rx, d_ry, 0))
            
            # ROTATION CENTER UPDATE:
            # We want to rotate around the CURRENT position of the object.
            # The object's center is at (target_tx, target_ty, 0) because we translated it there.
            # Using this as the center means the object rotates in place,
            # instead of swinging around the world origin (0,0,0).
            center_point = (target_tx, target_ty, 0)
            
            mesh.rotate(R, center=center_point)
            coords.rotate(R, center=center_point)
            vis.update_geometry(mesh)
            vis.update_geometry(coords)
            last_rot_x = target_rx
            last_rot_y = target_ry
            
        # 2. Scale
        if abs(target_scale - current_scale) > 1e-3:
            factor = target_scale / current_scale
            # Scale around the current center too
            center_point = (target_tx, target_ty, 0)
            
            mesh.scale(factor, center=center_point)
            coords.scale(factor, center=center_point)
            vis.update_geometry(mesh)
            vis.update_geometry(coords)
            current_scale = target_scale
            
        # 3. Pan
        d_tx = target_tx - last_trans_x
        d_ty = target_ty - last_trans_y
        
        if abs(d_tx) > 1e-4 or abs(d_ty) > 1e-4:
            mesh.translate((d_tx, d_ty, 0))
            coords.translate((d_tx, d_ty, 0))
            vis.update_geometry(mesh)
            vis.update_geometry(coords)
            last_trans_x = target_tx
            last_trans_y = target_ty

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.010)
        
    vis.destroy_window()
    print("Application Exit.")

if __name__ == "__main__":
    main()