# exercise.py
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables
exercise_duration = 30
start_time = time.time()
current_exercise_index = 0
exercise_sequence = ['squat', 'curl', 'situp', 'lunge', 'pushup']
exercise_counters = {ex: 0 for ex in exercise_sequence}
feedback = ""
state = "Down"
range_flag = True
halfway = False

def calc_angle(x, y, z):
    x, y, z = np.array(x), np.array(y), np.array(z)
    radians = np.arctan2(z[1]-y[1], z[0]-y[0]) - np.arctan2(x[1]-y[1], x[0]-y[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def recognise_squat(detection):
    global state, feedback, exercise_counters
    try:
        landmarks = detection.pose_landmarks.landmark
        
        # Get coordinates
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate angles
        left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)

        if left_knee_angle > 160 and right_knee_angle > 160:
            state = "Up"
            feedback = 'Good form, go down'
        elif left_knee_angle < 90 and right_knee_angle < 90 and state == "Up":
            state = "Down"
            exercise_counters['squat'] += 1
            feedback = "Good squat!"
    except Exception as e:
        print(f"Error in squat recognition: {e}")

def recognise_curl(detection):
    global state, feedback, exercise_counters, range_flag
    try:
        landmarks = detection.pose_landmarks.landmark
        
        # Get coordinates
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

        if left_angle > 160 and right_angle > 160:
            state = "Down"
            feedback = 'Good form, curl up'
            range_flag = True
        elif left_angle < 30 and right_angle < 30 and state == "Down":
            state = "Up"
            if range_flag:
                exercise_counters['curl'] += 1
                feedback = "Good curl!"
            range_flag = False
    except Exception as e:
        print(f"Error in curl recognition: {e}")

def recognise_situp(detection):
    global state, feedback, exercise_counters, range_flag, halfway
    try:
        landmarks = detection.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        # Calculate angle
        angle = calc_angle(shoulder, hip, knee)

        if angle < 45:
            state = "Up"
            halfway = True
            feedback = 'Good form, go down'
        elif angle > 150 and state == "Up":
            state = "Down"
            if halfway:
                exercise_counters['situp'] += 1
                feedback = "Good situp!"
            halfway = False
    except Exception as e:
        print(f"Error in situp recognition: {e}")

def recognise_lunge(detection):
    global state, feedback, exercise_counters
    try:
        landmarks = detection.pose_landmarks.landmark
        
        # Get coordinates
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Calculate angle
        lunge_angle = calc_angle(left_hip, left_knee, left_ankle)

        if lunge_angle > 160:
            state = "Up"
            feedback = 'Good form, go down'
        elif lunge_angle < 90 and state == "Up":
            state = "Down"
            exercise_counters['lunge'] += 1
            feedback = "Good lunge!"
    except Exception as e:
        print(f"Error in lunge recognition: {e}")

def recognise_pushup(detection):
    global state, feedback, exercise_counters
    try:
        landmarks = detection.pose_landmarks.landmark
        
        # Get coordinates
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angle
        angle = calc_angle(left_shoulder, left_elbow, left_wrist)

        if angle > 160:
            state = "Up"
            feedback = 'Good form, go down'
        elif angle < 90 and state == "Up":
            state = "Down"
            exercise_counters['pushup'] += 1
            feedback = "Good pushup!"
    except Exception as e:
        print(f"Error in pushup recognition: {e}")

def generate_frames():
    global start_time, current_exercise_index
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()
                elapsed_time = current_time - start_time
                time_left = max(0, exercise_duration - elapsed_time)

                if elapsed_time >= exercise_duration:
                    current_exercise_index = (current_exercise_index + 1) % len(exercise_sequence)
                    start_time = time.time()

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    current_exercise = exercise_sequence[current_exercise_index]
                    if current_exercise in exercise_functions:
                        exercise_functions[current_exercise](results)

                cv2.putText(image, f"Exercise: {exercise_sequence[current_exercise_index]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Time: {int(time_left)}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                print(f"Error in generate_frames: {e}")
                continue

    cap.release()

exercise_functions = {
    'squat': recognise_squat,
    'curl': recognise_curl,
    'situp': recognise_situp,
    'lunge': recognise_lunge,
    'pushup': recognise_pushup
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exercise_data')
def exercise_data():
    global current_exercise_index, start_time
    time_left = max(0, exercise_duration - (time.time() - start_time))
    return jsonify({
        'current_exercise': exercise_sequence[current_exercise_index],
        'feedback': feedback,
        'reps': [exercise_counters[ex] for ex in exercise_sequence],
        'time_left': int(time_left)
    })

if __name__ == '__main__':
    app.run(debug=True)