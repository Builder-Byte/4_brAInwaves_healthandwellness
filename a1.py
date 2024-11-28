import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables
exercise_duration = 30  # seconds per exercise
start_time = time.time()
current_exercise = 1
exercise_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
feedback = ""
state = "Down"

def calc_angle(x, y, z):
    x, y, z = np.array(x), np.array(y), np.array(z)
    radians = np.arctan2(z[1]-y[1], z[0]-y[0]) - np.arctan2(x[1]-y[1], x[0]-y[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def recognise_pushup(landmarks):
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

        global state, feedback
        
        if left_angle > 160 and right_angle > 160:
            state = "Up"
            feedback = 'Good form, go down'
        
        if left_angle < 90 and right_angle < 90 and state == "Up":
            state = "Down"
            exercise_counters[5] += 1
            feedback = "Good pushup!"
            
        return left_angle, right_angle
        
    except:
        return 180, 180

def recognise_squat(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
        # Get hip, knee and ankle points
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles
        left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)

        left_angle.append(int(left_knee_angle))
        right_angle.append(int(right_knee_angle))

        if left_knee_angle > 160 and right_knee_angle > 160:
            state = "Up"
            feedback = 'Good form, go down'

        if left_knee_angle < 90 and right_knee_angle < 90 and state == "Up":
            state = "Down"
            exercise_counters[1] += 1
            feedback = "Good squat!"
    except:
        left_angle.append(180)
        right_angle.append(180)

def recognise_curl(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
        # Get shoulder, elbow and wrist points
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

        if left_elbow_angle > 160 and right_elbow_angle > 160:
            state = "Down"
            feedback = 'Good form, curl up'

        if left_elbow_angle < 30 and right_elbow_angle < 30 and state == "Down":
            state = "Up"
            exercise_counters[2] += 1
            feedback = "Good curl!"
    except:
        left_angle.append(180)
        right_angle.append(180)

def recognise_situp(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
        # Get shoulder, hip and knee points
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        # Calculate angle
        trunk_angle = calc_angle(left_shoulder, left_hip, left_knee)

        if trunk_angle < 45:
            state = "Up"
            feedback = 'Good form, go down'

        if trunk_angle > 150 and state == "Up":
            state = "Down"
            exercise_counters[3] += 1
            feedback = "Good situp!"
    except:
        left_angle.append(180)
        right_angle.append(180)

def generate_frames():
    global start_time, current_exercise, feedback
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            time_left = max(0, exercise_duration - elapsed_time)

            if time_left == 0:
                current_exercise = (current_exercise % 5) + 1
                start_time = time.time()
                feedback = f"Switching to exercise {current_exercise}"

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                recognise_pushup(results.pose_landmarks.landmark)

            # Add exercise info to frame
            cv2.putText(image, f"Exercise: {current_exercise}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Time left: {int(time_left)}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exercise_data')
def exercise_data():
    global current_exercise, start_time, exercise_duration
    time_left = max(0, exercise_duration - (time.time() - start_time))
    return jsonify({
        'feedback': feedback,
        'reps': list(exercise_counters.values()),
        'time_left': int(time_left),
        'current_exercise': current_exercise
    })

if __name__ == '__main__':
    app.run(debug=True)