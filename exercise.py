import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calc_angle(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def recognise_squat(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left = calc_angle(left_hip, left_knee, left_heel)
        right = calc_angle(right_hip, right_knee, right_heel)
        left_angle.append(int(left))
        right_angle.append(int(right))
        shoulder_dist = left_shoulder[0] - right_shoulder[0]
        knee_dist = left_knee[0] - right_knee[0]
        if shoulder_dist - knee_dist > 0.04:
            feedback = 'Open up your knees further apart to shoulder width!'
        else:
            feedback = ''
        if left > 170 and right > 170:
            state = "Up"
        if left < 165 and right < 165:
            feedback = 'Almost there... lower until height of hips!'
        if left < 140 and right < 140 and state == "Up":
            state = "Down"
            exercise_counters[1] += 1
        if state == "Down":
            feedback = 'Good rep!'
    except:
        left_angle.append(180)
        right_angle.append(180)


def recognise_curl(detection):
    global exercise_counters, state, feedback, range_flag, left_angle, right_angle
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
        left_angle.append(int(left_elbow_angle))
        right_angle.append(int(right_elbow_angle))
        if left_elbow_angle > 160 and right_elbow_angle > 160:
            if not range_flag:
                feedback = 'Did not curl completely.'
            else:
                feedback = 'Good rep!'
            state = 'Down'
        elif (left_elbow_angle > 50 and right_elbow_angle > 50) and state == 'Down':
            range_flag = False
            feedback = ''
        elif (left_elbow_angle < 30 and right_elbow_angle < 30) and state == 'Down':
            state = 'Up'
            feedback = ''
            range_flag = True
            exercise_counters[2] += 1
    except:
        left_angle.append(180)
        right_angle.append(180)


def recognise_situp(detection):
    global exercise_counters, state, feedback, range_flag, halfway, body_angles
    try:
        landmarks = detection.pose_landmarks.landmark
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        angle_knee = calc_angle(left_hip, left_knee, left_heel)
        angle_body = calc_angle(left_shoulder, left_hip, left_knee)
        body_angles.append(int(angle_body))
        if (angle_body < 80 and angle_body > 50) and state == "Down":
            halfway = True
        if angle_body < 40 and state == "Down":
            state = "Up"
            range_flag = True
        if angle_body > 90 and angle_knee < 60:
            state = "Down"
            if halfway:
                if range_flag:
                    exercise_counters[3] += 1
                    feedback = "Good repetition!"
                else:
                    feedback = "Did not perform sit up completely."
                range_flag = False
                halfway = False
        if angle_knee > 70:
            feedback = "Keep legs tucked in closer"
    except:
        body_angles.append(180)


def recognise_lunge(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
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

        left_lunge_angle = calc_angle(left_hip, left_knee, left_ankle)
        right_lunge_angle = calc_angle(right_hip, right_knee, right_ankle)

        left_angle.append(int(left_lunge_angle))
        right_angle.append(int(right_lunge_angle))

        if left_lunge_angle > 160 and right_lunge_angle > 160:
            state = "Up"
            feedback = ''

        if (left_lunge_angle < 100 and right_lunge_angle < 100) and state == "Up":
            state = "Down"
            exercise_counters[4] += 1
            feedback = "Good lunge!"
    except:
        left_angle.append(180)
        right_angle.append(180)


def recognise_pushup(detection):
    global exercise_counters, state, feedback, left_angle, right_angle
    try:
        landmarks = detection.pose_landmarks.landmark
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

        left_elbow_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

        left_angle.append(int(left_elbow_angle))
        right_angle.append(int(right_elbow_angle))

        if left_elbow_angle > 160 and right_elbow_angle > 160:
            state = "Up"
            feedback = ''

        if (left_elbow_angle < 90 and right_elbow_angle < 90) and state == "Up":
            state = "Down"
            exercise_counters[5] += 1
            feedback = "Good pushup!"
    except:
        left_angle.append(180)
        right_angle.append(180)


def plot_viz(exercise_choice):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.title('Exercise Reps')
    exercise_labels = ['', 'Squat', 'Arm Curl', 'Sit-up', 'Lunge', 'Pushup']
    plt.bar(exercise_labels[1:], exercise_counters[1:])
    plt.ylabel('Number of Reps')

    # Individual exercise angle plots
    if exercise_choice == 1:
        plt.subplot(2, 3, 2)
        plt.plot(frames, left_angle, '-', color='red', label='Left Knee Angle')
        plt.plot(frames, right_angle, '-', color='blue', label='Right Knee Angle')
        plt.axhline(y=140, color='g', linestyle='--')
        plt.title('Squat Angles')
        plt.xlabel('Frames')
        plt.ylabel('Angle')
        plt.legend()

    elif exercise_choice == 2:
        plt.subplot(2, 3, 2)
        plt.plot(frames, left_angle, '-', color='red', label='Left Arm Angle')
        plt.plot(frames, right_angle, '-', color='blue', label='Right Arm Angle')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('Arm Curl Angles')
        plt.xlabel('Frames')
        plt.ylabel('Angle')
        plt.legend()

    elif exercise_choice == 3:
        plt.subplot(2, 3, 2)
        plt.plot(frames, body_angles, '-', color='red', label='Body Angle')
        plt.axhline(y=40, color='g', linestyle='--')
        plt.title('Sit-up Angles')
        plt.xlabel('Frames')
        plt.ylabel('Angle')
        plt.legend()

    elif exercise_choice == 4:
        plt.subplot(2, 3, 2)
        plt.plot(frames, left_angle, '-', color='red', label='Left Lunge Angle')
        plt.plot(frames, right_angle, '-', color='blue', label='Right Lunge Angle')
        plt.axhline(y=100, color='g', linestyle='--')
        plt.title('Lunge Angles')
        plt.xlabel('Frames')
        plt.ylabel('Angle')
        plt.legend()

    elif exercise_choice == 5:
        plt.subplot(2, 3, 2)
        plt.plot(frames, left_angle, '-', color='red', label='Left Arm Angle')
        plt.plot(frames, right_angle, '-', color='blue', label='Right Arm Angle')
        plt.axhline(y=90, color='g', linestyle='--')
        plt.title('Pushup Angles')
        plt.xlabel('Frames')
        plt.ylabel('Angle')
        plt.legend()

    # Add text with overall feedback and how to continue/exit
    plt.subplot(2, 3, (3, 6))
    plt.axis('off')

    feedback_text = "Exercise Performance Summary:\n\n"
    exercise_labels = ['', 'Squats', 'Arm Curls', 'Sit-ups', 'Lunges', 'Pushups']

    total_reps = sum(exercise_counters[1:])
    feedback_text += f"Total Repetitions: {total_reps}\n\n"

    for i in range(1, 6):
        feedback_text += f"{exercise_labels[i]}: {exercise_counters[i]} reps\n"

    # Performance evaluation
    if total_reps == 0:
        overall_feedback = "No exercises completed. Keep trying!"
    elif total_reps < 10:
        overall_feedback = "Good start! Keep practicing."
    elif total_reps < 20:
        overall_feedback = "Nice work! You're making progress."
    elif total_reps < 30:
        overall_feedback = "Great job! You're getting stronger."
    else:
        overall_feedback = "Excellent performance! You're a fitness champion!"

    feedback_text += f"\nOverall Feedback:\n{overall_feedback}\n"

    # Muscle group suggestions based on exercises
    muscle_groups = {
        1: "Quadriceps, Glutes, Hamstrings",
        2: "Biceps, Forearms",
        3: "Core, Abdominal Muscles",
        4: "Quadriceps, Glutes, Calves",
        5: "Chest, Triceps, Core"
    }

    feedback_text += "\nMuscle Groups Worked:\n"
    worked_muscles = set()
    for i in range(1, 6):
        if exercise_counters[i] > 0:
            worked_muscles.update(muscle_groups[i].split(", "))

    feedback_text += ", ".join(worked_muscles)



    plt.text(0.1, 0.5, feedback_text, fontsize=10,
             verticalalignment='center',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def main():
    global exercise_counters, state, feedback, range_flag, halfway, left_angle, right_angle, body_angles, frames

    # Initialize global variables
    exercise_counters = [0, 0, 0, 0, 0, 0]
    state = "Up"
    feedback = ""
    range_flag = True
    halfway = False
    left_angle = []
    right_angle = []
    body_angles = []
    frames = []

    # Exercise recognition function dictionary
    exercise_functions = {
        1: recognise_squat,
        2: recognise_curl,
        3: recognise_situp,
        4: recognise_lunge,
        5: recognise_pushup
    }

    # Predefined exercise sequence
    exercise_sequence = [1, 2, 3, 4, 5]
    current_exercise_index = 0
    exercise_choice = exercise_sequence[current_exercise_index]

    # Video capture setup
    cap = cv2.VideoCapture(0)

    # MediaPipe Pose setup
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        start_time = time.time()
        exercise_duration = 30  # 30 seconds per exercise

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check if current exercise time has elapsed
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_left = max(0, exercise_duration - elapsed_time)

            if elapsed_time >= exercise_duration:
                # Move to next exercise
                current_exercise_index = (current_exercise_index + 1) % len(exercise_sequence)
                exercise_choice = exercise_sequence[current_exercise_index]

                # Reset variables for new exercise
                state = "Up"
                feedback = ""
                range_flag = True
                halfway = False
                left_angle.clear()
                right_angle.clear()
                body_angles.clear()
                frames.clear()
                exercise_counters[exercise_choice] = 0

                # Reset start time
                start_time = time.time()

            # Convert frame to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Pose detection
            results = pose.process(image)

            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detected landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Call the appropriate exercise recognition function
                exercise_functions[exercise_choice](results)
                frames.append(frame_count)
                frame_count += 1

            # Display exercise information
            cv2.putText(
                image,
                f"Exercise: {['', 'Squat', 'Curl', 'Sit-up', 'Lunge', 'Pushup'][exercise_choice]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.putText(
                image,
                f"Reps: {exercise_counters[exercise_choice]}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.putText(
                image,
                f"Time Left: {int(time_left)}s",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                image,
                feedback,
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Display frame
            cv2.imshow('Fitness Tracker', image)

            # Quit application with 'q'
            if cv2.waitKey(10) == ord('q'):
                plot_viz(exercise_choice)
                break

        cap.release()
        cv2.destroyAllWindows()


# Start the program
if __name__ == "__main__":
    main()
