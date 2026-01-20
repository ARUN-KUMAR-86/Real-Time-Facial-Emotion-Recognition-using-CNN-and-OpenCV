# Real-Time-Facial-Emotion-Recognition-using-CNN-and-OpenCV
A deep learning–based facial emotion recognition system using CNN, TensorFlow, and OpenCV. The model is trained on grayscale facial images to classify emotions and performs real-time emotion detection via webcam with confidence scores, enabling applications in human–computer interaction and behavioral analysis

# Emotion Detection using CNN

This project detects human emotions in real time using a webcam.
A CNN model is trained on facial images and used to classify emotions
such as happy, sad, angry, and neutral.

## Flow
Dataset → CNN Training → Model Saved → Webcam Emotion Detection

## Run
```bash
pip install -r requirements.txt
python src/emotion_detection.py
