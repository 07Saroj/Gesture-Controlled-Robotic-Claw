import cv2 
import mediapipe as mp 
import numpy as np 
import asyncio 
import websockets 
import time 
 
# Initialize MediaPipe Hands with GPU acceleration 
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 
hands = mp_hands.Hands( 
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    model_complexity=1  # Balanced performance/accuracy 
) 
 
# WebSocket setup 
RASPBERRY_PI_IP = "192.168.176.81" 
PORT = 8765 
WS_URL = f"ws://{RASPBERRY_PI_IP}:{PORT}" 
 
# Performance tracking 
prev_time = 0 
fps_list = [] 
 
# Hand control variables 
ACTIVE_HAND = "right" 
LAST_SWITCH_TIME = 0 
SWITCH_COOLDOWN = 2 
MESSAGE_DISPLAY_TIME = 3 
message_display_start = 0 
show_switch_message = False 
 
# Exponential Moving Average Smoothing 
SMOOTHING_ALPHA = 0.3 
smoothed_openness = None 
smoothed_x_angle = None 
smoothed_y_angle = None 
 
def draw_hand_landmarks(frame, hand_landmarks): 
    """Draw landmarks with gradient colors""" 
    for i, landmark in enumerate(hand_landmarks.landmark): 
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]) 
        color = (i * 12, 255 - i * 12, 255) 
        cv2.circle(frame, (x, y), 6, color, -1) 
 
def draw_modern_overlay(frame, fps): 
    """Draw FPS and info overlay""" 
    h, w = frame.shape[:2] 
    overlay = frame.copy() 
    cv2.rectangle(overlay, (10, h-100), (w-10, h-10), (0, 0, 0), -1) 
    alpha = 0.4 
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) 
     
    fps_list.append(fps) 
    if len(fps_list) > 30: 
        fps_list.pop(0) 
    avg_fps = sum(fps_list) / len(fps_list) 
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) 
    return frame 
 
def is_l_gesture(landmarks): 
    """Detect L-shaped hand gesture""" 
    wrist = landmarks[0] 
    thumb_tip = landmarks[4] 
    index_tip = landmarks[8] 
    middle_tip = landmarks[12] 
    ring_tip = landmarks[16] 
    pinky_tip = landmarks[20] 
     
    # Check if fingers are closed 
    closed_fingers = all( 
        np.linalg.norm(np.array([f.x, f.y]) - np.array([wrist.x, wrist.y])) < 0.1 
        for f in [middle_tip, ring_tip, pinky_tip] 
    ) 
     
    # Calculate angle between thumb and index 
    thumb_vec = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y]) 
    index_vec = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y]) 
    angle = np.degrees(np.arccos( 
        np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * 
np.linalg.norm(index_vec)) 
    )) 
    return closed_fingers and (50 < angle < 100) 
 
def compute_hand_openness(landmarks): 
    """Calculate hand openness percentage""" 
    finger_tips = [4, 8, 12, 16, 20] 
    distances = [] 
    for tip in finger_tips: 
        dist = np.linalg.norm(np.array([landmarks[tip].x, landmarks[tip].y]) -  
                             np.array([landmarks[0].x, landmarks[0].y])) 
        distances.append(dist) 
    min_open, max_open = 0.05, 0.3 
    avg_distance = np.mean(distances) 
    return np.clip((avg_distance - min_open) / (max_open - min_open), 0, 1) * 100 
 
def compute_angles(landmarks, frame_width, frame_height): 
    """Calculate X and Y servo angles""" 
    x_angle = np.clip((landmarks[0].x * frame_width / frame_width) * 180, 0, 180) 
    y_angle = np.clip(90 + ((landmarks[0].y * frame_height / frame_height) * 90), 90, 180) 
    return x_angle, y_angle 
 
async def send_command(openness, x_angle, y_angle): 
    """Send command to Raspberry Pi""" 
    async with websockets.connect(WS_URL) as websocket: 
        command_data = f"{openness:.2f},{x_angle:.2f},{y_angle:.2f}" 
        await websocket.send(command_data) 
        print(f"Sent: {command_data}") 
 
# Main processing loop 
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
 
previous_openness = None 
previous_x_angle = None 
previous_y_angle = None 
 
try: 
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if not ret: 
            break 
 
        # Mirror and convert to RGB 
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
         
        # Process with MediaPipe 
        results = hands.process(rgb_frame) 
        current_time = time.time() 
         
        if results.multi_hand_landmarks: 
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                results.multi_handedness): 
                landmarks = hand_landmarks.landmark 
                hand_label = handedness.classification[0].label.lower() 
                 
                # Check for L-gesture to switch hands 
                if is_l_gesture(landmarks) and (current_time - LAST_SWITCH_TIME) > SWITCH_COOLDOWN: 
                    ACTIVE_HAND = "left" if ACTIVE_HAND == "right" else "right" 
                    LAST_SWITCH_TIME = current_time 
                    message_display_start = current_time 
                    show_switch_message = True 
                    print(f"Switched dominant hand to: {ACTIVE_HAND}") 
                 
                if hand_label == ACTIVE_HAND: 
                    # Draw landmarks 
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
                    draw_hand_landmarks(frame, hand_landmarks) 
                     
                    # Compute metrics 
                    openness = compute_hand_openness(landmarks) 
                    x_angle, y_angle = compute_angles(landmarks, frame.shape[1], frame.shape[0]) 
                     
                    # Apply smoothing 
                    if smoothed_openness is None: 
                        smoothed_openness = openness 
                        smoothed_x_angle = x_angle 
                        smoothed_y_angle = y_angle 
                    else: 
                        smoothed_openness = SMOOTHING_ALPHA * openness + (1 - SMOOTHING_ALPHA) * smoothed_openness 
                        smoothed_x_angle = SMOOTHING_ALPHA * x_angle + (1 - SMOOTHING_ALPHA) * smoothed_x_angle 
                        smoothed_y_angle = SMOOTHING_ALPHA * y_angle + (1 - SMOOTHING_ALPHA) * smoothed_y_angle 
                     
                    # Send command if significant change 
                    if (abs(smoothed_openness - (previous_openness or 0))) > 2 or (abs(smoothed_x_angle - (previous_x_angle or 0))) > 2 or (abs(smoothed_y_angle - (previous_y_angle or 0))) > 2: 
                        asyncio.run(send_command(smoothed_openness, smoothed_x_angle, smoothed_y_angle)) 
                        previous_openness = smoothed_openness 
                        previous_x_angle = smoothed_x_angle 
                        previous_y_angle = smoothed_y_angle 
                     
                    # Display values 
                    cv2.putText(frame, f"Active: {ACTIVE_HAND.upper()}", (10, 30), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2) 
                    cv2.putText(frame, f"Openness: {smoothed_openness:.2f}%", (10, 70), 
                               cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2) 
                    cv2.putText(frame, f"X Servo: {smoothed_x_angle:.2f}degree", (10, 110), 
                               cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2) 
                    cv2.putText(frame, f"Y Servo: {smoothed_y_angle:.2f}degree", (10, 150), 
                               cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2) 
 
        # Calculate and display FPS 
        curr_time = time.time() 
        fps = 1/(curr_time - prev_time) 
        prev_time = curr_time 
        frame = draw_modern_overlay(frame, fps) 
         
        # Show switch message if needed 
        if show_switch_message and (current_time - message_display_start) < MESSAGE_DISPLAY_TIME: 
            cv2.putText(frame, f"Switched to {ACTIVE_HAND} hand!", (495, 650), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
        else: 
            show_switch_message = False 
         
        # Display 
        cv2.imshow("Hand Tracking", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
 
finally: 
    cap.release() 
    cv2.destroyAllWindows()