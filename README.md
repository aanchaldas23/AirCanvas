#  Air Canvas

This project is a virtual paint application that uses a webcam to track hand movements with Mediapipe's Hand Tracking module. The application allows users to paint on a virtual canvas by raising their index finger and selecting colors through on-screen buttons.

---

## Features

- **Hand Tracking**: Uses Mediapipe to detect and track the user's hand and fingers.
- **Drawing on a Canvas**: Enables painting by moving the index finger across the screen.
- **Color Selection**: Provides buttons for selecting colors (Blue, Green, Red, Yellow).
- **Clear Canvas**: A button to reset the canvas and start fresh.
- **Real-Time Feedback**: Visualizes hand landmarks on the webcam feed.

---

## Prerequisites

- Python 3.7 or later
- OpenCV for image processing
- Mediapipe for hand tracking
- NumPy for matrix operations

---

## How It Works

1. **Hand Detection**: The application detects your hand and identifies the landmarks for your fingers using Mediapipe's `Hands` module.
2. **Drawing Mechanism**:
   - The tip of the index finger is used as the brush.
   - When the index finger is raised and moved across the screen, a line is drawn on the canvas.
3. **Color Selection**: By moving the index finger over specific color buttons on the top of the screen, you can switch between colors.
4. **Clear Button**: To erase the canvas, hover your finger over the "CLEAR" button.

---

## Usage Instructions

1. Run the program and ensure your webcam is enabled.
2. Raise your index finger to start drawing on the virtual canvas.
3. Hover your index finger over the color buttons to switch colors.
4. To clear the canvas, hover your finger over the "CLEAR" button.
5. Press `q` to quit the program.

