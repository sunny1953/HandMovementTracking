import numpy as np
import cv2
import mediapipe as mp
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    hands = mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.9,
        max_num_hands=2)
    
    # Drawing specs for hand landmarks
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    
    cap = cv2.VideoCapture(0)
    prev_points = {}
    points_queues = {}
    
    # Adjusted smoothing parameters
    smoothing_factor = 0.3
    
    # Create canvas for storing drawings
    _, first_frame = cap.read()
    canvas = np.zeros_like(first_frame)
    
    while cap.isOpened():
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        
        image = cv2.cvtColor(cv2.resize(image, (640, 480)), cv2.COLOR_BGR2RGB)
        canvas = cv2.resize(canvas, (640, 480))
        image.flags.writeable = False
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results_hand.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
                
                # Get index finger tip position
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = image.shape
                raw_point = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                
                # Initialize queue for this hand if it doesn't exist
                if idx not in points_queues:
                    points_queues[idx] = deque(maxlen=5)
                
                # Add point to queue for averaging
                points_queues[idx].append(raw_point)
                
                # Calculate smoothed point
                if len(points_queues[idx]) >= 3:
                    avg_x = int(sum(p[0] for p in points_queues[idx]) / len(points_queues[idx]))
                    avg_y = int(sum(p[1] for p in points_queues[idx]) / len(points_queues[idx]))
                    current_point = (avg_x, avg_y)
                    
                    # Additional smoothing with previous point
                    if idx in prev_points and prev_points[idx] is not None:
                        smooth_x = int(prev_points[idx][0] * smoothing_factor + current_point[0] * (1 - smoothing_factor))
                        smooth_y = int(prev_points[idx][1] * smoothing_factor + current_point[1] * (1 - smoothing_factor))
                        current_point = (smooth_x, smooth_y)
                        
                        # Only draw if movement is significant enough
                        distance = np.sqrt((smooth_x - prev_points[idx][0])**2 + (smooth_y - prev_points[idx][1])**2)
                        if distance > 2:
                            # Use different colors for each hand
                            color = (0, 255, 0) if idx == 0 else (255, 0, 0)
                            canvas_color = (0, 0, 255) if idx == 0 else (255, 0, 0)
                            cv2.line(image, prev_points[idx], current_point, color, 2)
                            cv2.line(canvas, prev_points[idx], current_point, canvas_color, 2)
                    
                    prev_points[idx] = current_point
        else:
            prev_points.clear()
            points_queues.clear()
        
        combined_image = cv2.addWeighted(image, 1.0, canvas, 1.0, 0)
        cv2.imshow("Res", rescale_frame(combined_image, percent=130))
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            canvas = np.zeros_like(canvas)
            points_queues.clear()
    
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
