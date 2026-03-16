# traffic-accident-detection

Traffic Accident Detection using Deep Learning - CADP Dataset

Course: Data Analytics Engineering — Northeastern University Vancouver
Dataset: CADP (Car Accident Detection and Prediction)

Project Overview
This project builds and compares three deep learning models to automatically detect traffic accidents in video footage using the CADP dataset, which contains 1,416 video segments collected from CCTV traffic cameras.

Models

ResNet50 + Logistic Regression — Baseline model using pretrained CNN features
3D CNN — Learns spatial and temporal features simultaneously
CNN-LSTM — CNN extracts frame features, LSTM learns temporal patterns


Dataset

1,416 video folders with pre-extracted JPG frames
2,754 total samples (2 sequences per video — accident and normal)
Perfectly balanced: 1,377 accident and 1,377 normal samples
Split: 70% train, 15% validation, 15% test
Dataset: https://ankitshah009.github.io/accident_forecasting_traffic_camera


How to Run

Download the CADP dataset and upload extracted frames to Google Drive at /content/drive/MyDrive/extracted_frames/
Open the notebook in Google Colab
Mount Google Drive
Run cells in order


Dependencies
tensorflow, numpy, opencv-python, scikit-learn, tqdm

Repository Structure

traffic_accident_detection.ipynb — Main notebook with all code
README.md — Project description


Results
ResNet50 + Logistic Regression: 49% test accuracy
3D CNN: In Progress
CNN-LSTM: In Progress

Reference
Shah, A.P., Lamare, J.B., Nguyen-Anh, T. and Hauptmann, A., 2018. CADP: A novel dataset for CCTV traffic camera based accident analysis. IEEE AVSS 2018.
