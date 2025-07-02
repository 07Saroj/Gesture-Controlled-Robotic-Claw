# Gesture-Controlled-Robotic-Claw
# Gesture Controlled Robotic Claw using Raspberry Pi and MediaPipe

This project enables a robotic claw + arm to be controlled in real-time using hand gestures via computer vision and a Raspberry Pi.

## Features
- Real-time gesture tracking using MediaPipe
- X/Y motion and claw control mapped from hand position and openness
- WebSocket communication between client and Raspberry Pi
- L-Gesture to switch dominant hand

## Technologies Used
- Python (MediaPipe, OpenCV, asyncio, websockets)
- Raspberry Pi 4 with PCA9685 Servo Driver
- Servo motors (MG996R)
