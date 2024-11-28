# AI Fitness Trainer

## Overview
AI Fitness Trainer is an interactive web application that uses computer vision to track and count exercise repetitions in real-time. The application leverages MediaPipe's pose estimation to recognize and validate different types of exercises.

## Features
- Real-time video feed with exercise tracking
- Supports multiple exercises:
  - Squats
  - Curls
  - Situps
  - Lunges
  - Pushups
- Automatic exercise progression
- Repetition counting
- Form feedback
- Time-based exercise sessions

## Prerequisites
- Python 3.8+
- Camera-enabled device

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Builder-Byte/4_brAInwaves_healthandwellness.git
cd 4_brAInwaves_healthandwellness
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python exercise.py
```

5. Open a web browser and navigate to `http://localhost:5000`

## How It Works
The application uses MediaPipe's pose estimation to track body landmarks and recognize different exercise movements. Each exercise has a specific recognition algorithm that counts repetitions based on body angle and position.

## Supported Exercises
- **Squats**: Tracked by knee and hip angles
- **Curls**: Tracked by elbow and shoulder angles
- **Situps**: Tracked by shoulder, hip, and knee angles
- **Lunges**: Tracked by hip, knee, and ankle angles
- **Pushups**: Tracked by shoulder, elbow, and wrist angles

## Technologies Used
- Flask
- OpenCV
- MediaPipe
- Python

## Limitations
- Requires good lighting and clear camera view
- Best used with full-body visibility
- Accuracy depends on camera quality and positioning

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License
