from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    radians = np.arctan2(z[1]-y[1], z[0]-y[0]) - np.arctan2(x[1]-y[1], x[0]-y[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class ExerciseTracker:
    def __init__(self):
        self.counter = 0
        self.state = 'Down'
        self.feedback = ''

    def recognise_curl(self, detection):
        try:
            landmarks = detection.pose_landmarks.landmark
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            if left_elbow_angle > 160 and right_elbow_angle > 160:
                self.feedback = 'Good rep!'
                self.state = 'Down'
            elif left_elbow_angle < 30 and right_elbow_angle < 30 and self.state == 'Down':
                self.state = 'Up'
                self.counter += 1
        except:
            self.feedback = 'No pose detected'

    def recognise_squat(self, detection):
        try:
            landmarks = detection.pose_landmarks.landmark
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

            left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)

            if left_knee_angle > 160 and right_knee_angle > 160:
                self.feedback = 'Good rep!'
                self.state = 'Up'
            elif left_knee_angle < 90 and right_knee_angle < 90 and self.state == 'Up':
                self.state = 'Down'
                self.counter += 1
        except:
            self.feedback = 'No pose detected'

    def recognise_situp(self, detection):
        try:
            landmarks = detection.pose_landmarks.landmark
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee)

            if left_hip_angle > 160 and right_hip_angle > 160:
                self.feedback = 'Good rep!'
                self.state = 'Down'
            elif left_hip_angle < 90 and right_hip_angle < 90 and self.state == 'Down':
                self.state = 'Up'
                self.counter += 1
        except:
            self.feedback = 'No pose detected'

tracker = ExerciseTracker()

def generate_frames(user_choice):
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            detection = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if detection.pose_landmarks:
                if user_choice == 1:
                    tracker.recognise_squat(detection)
                elif user_choice == 2:
                    tracker.recognise_curl(detection)
                elif user_choice == 3:
                    tracker.recognise_situp(detection)

            mp_drawing.draw_landmarks(image, detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    user_choice = int(request.args.get('exercise', 1))
    return Response(generate_frames(user_choice),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_feedback')
def get_feedback():
    return jsonify({
        "feedback": tracker.feedback,
        "counter": tracker.counter
    })

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    tracker.counter = 0
    tracker.state = 'Down'
    tracker.feedback = ''
    return jsonify({"message": "Counter reset successfully"})

if __name__ == '__main__':
    app.run(debug=True)