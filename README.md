# Traffic Accident Detection Using Deep Learning

Overview
Road traffic accidents remain one of the leading causes of preventable fatalities globally, yet the infrastructure to detect them automatically and in real time remains underdeveloped. Modern urban surveillance systems produce continuous video streams that far exceed the capacity of human monitoring. This project investigates whether deep learning models trained on short video sequences can reliably distinguish accident events from normal traffic conditions, without requiring manual per-frame annotation or complex multi-stage detection pipelines.
Three architectures are designed, trained, and evaluated on the CADP dataset: a transfer learning baseline using ResNet50 and Logistic Regression, a CNN-LSTM hybrid that incorporates sequential modeling, and a custom 3D Convolutional Neural Network that learns spatiotemporal features end-to-end.

Problem Statement
Task: Binary classification of video segments into Accident (1) or Normal (0)
Input: A sequence of 16 frames extracted from a CCTV traffic camera recording
Output: A binary label — Accident (1) or Normal (0)
Core difficulty: Traffic accidents are low-frequency, visually subtle events that develop over short time windows. Effective detection requires a model that jointly captures both the spatial content of individual frames and the temporal dynamics across the sequence.

Dataset
CADP — Car Accident Detection and Prediction
Shah et al., IEEE AVSS 2018 · Dataset Page
Total Video Folders — 1,416
Frame Format — Pre-extracted JPG
Total Samples — 2,754
Accident Samples — 1,377 (50%)
Normal Samples — 1,377 (50%)
Train / Validation / Test — 70% / 15% / 15%
Labeling: The CADP dataset does not contain normal-only videos — every recording includes at least one accident event occurring near the beginning of the clip. To construct a balanced training set, two non-overlapping sequences are drawn from each video. Frames from the initial 30% of each video are assigned an accident label, while frames from the remaining 70% are assigned a normal label. This partitioning is grounded in the dataset's documented accident onset ratio of 0.01, consistent with the negative mining strategy described in the original paper.

Models
Model 1 — ResNet50 + Logistic Regression (Baseline)
ResNet50 pretrained on ImageNet is used as a fixed feature extractor. For each video, 16 frames are passed through the network to produce 2048-dimensional feature vectors, which are then averaged into a single representation and classified using Logistic Regression. A second experiment partially unfreezes the last 10 layers of ResNet50, though gradient flow remains disconnected from the classifier. Both configurations achieve 49% test accuracy, consistent with random guessing on a balanced binary task. The fundamental limitation is that temporal averaging collapses the sequential structure of the input, leaving no signal from which the classifier can learn event dynamics.

Model 2 — CNN-LSTM
Two configurations are evaluated. The first pairs a lightweight two-layer convolutional network with an LSTM(128) unit, operating on 64×64 frame inputs. The shallow feature extractor produces representations of insufficient quality, and the model converges at 50% accuracy. The second configuration replaces the shallow CNN with pre-extracted ResNet50 features fed as a sequence into an LSTM(256) with 2.3 million total parameters. This model also achieves 50% and collapses to predicting the Normal class exclusively. The underlying issue is that image-pretrained features encode static visual appearance rather than inter-frame motion, and accident frames are not visually distinguishable from normal frames on a per-frame basis.

Model 3 — 3D CNN (Best Model)
The 3D CNN treats the 16-frame input as a volumetric signal and applies three-dimensional convolutions that operate simultaneously across spatial and temporal dimensions. This allows the model to learn motion-sensitive filters without relying on pre-extracted features or a separate sequential module.
Input (16, 112, 112, 3) — 16-frame video volume
Conv3D(32) + BatchNorm + MaxPool3D — Low-level spatiotemporal feature extraction
Conv3D(64) + BatchNorm + MaxPool3D — Mid-level motion pattern learning
Conv3D(128) + BatchNorm + GlobalAvgPool — High-level abstract representations
Dense(256) + Dropout(0.5) — Fully connected layer with regularization
Dense(1) + Sigmoid — Binary classification output
Training Configuration: Optimizer: Adam, Learning Rate: 0.0001, Loss: Binary Crossentropy, Epochs: 10, Batch Size: 8, Total Parameters: 313,000

Results
ResNet50 + LR (frozen) — 49%
ResNet50 + LR (unfrozen) — 49%
CNN-LSTM (shallow CNN) — 50%
CNN-LSTM (ResNet50 + LSTM) — 50%
3D CNN — 70%
3D CNN Classification Report:
Normal — Precision: 0.66, Recall: 0.81, F1-Score: 0.73
Accident — Precision: 0.75, Recall: 0.58, F1-Score: 0.66
Confusion Matrix:
True Normal — Predicted Normal: 168, Predicted Accident: 39
True Accident — Predicted Normal: 87, Predicted Accident: 120

How to Run
This project was developed and executed on Google Colab with GPU acceleration. Local execution requires a compatible GPU and sufficient system memory.
Step 1: Download the CADP dataset and place the extracted frame folders into Google Drive at /content/drive/MyDrive/extracted_frames/
Step 2: Open deep_learning_project.ipynb in Google Colab
Step 3: Mount Google Drive when prompted at the start of the notebook
Step 4: Execute all cells sequentially — dataset preparation, model training, and evaluation are implemented as a single end-to-end pipeline

Dependencies
tensorflow 2.12.0 or higher
numpy 1.23.0 or higher
opencv-python 4.7.0 or higher
scikit-learn 1.2.0 or higher
tqdm 4.65.0 or higher
matplotlib 3.7.0 or higher

Repository Structure
deep_learning_project.ipynb — Main notebook containing all model implementations and evaluation
requirements.txt — Python package dependencies with pinned versions
README.md — Project documentation
.gitignore — Excludes large data files, model checkpoints, and cache directories

Discussion
The results demonstrate that standard image classification features are inadequate for video-based accident detection. Models that operate on static frame representations — whether through feature averaging or sequential processing of appearance descriptors — fail to exceed chance-level performance on this task. The 3D CNN succeeds because its convolutional filters span the temporal dimension directly, enabling the model to detect motion discontinuities and dynamic scene changes that are characteristic of collision events. The 21 percentage point improvement over all baselines indicates that the choice of spatiotemporal representation is the primary factor governing performance on this dataset, not dataset size or class distribution.

Future Work
Incorporating optical flow as an additional input modality would provide the model with explicit velocity information, potentially improving sensitivity to sudden motion changes. Replacing the custom 3D CNN with architectures pretrained on large-scale video datasets such as Kinetics — including C3D, I3D, or SlowFast — would provide stronger initial representations. More precise frame labeling using the exact accident timestamps available in the CADP annotations could reduce label noise in the training set. Standard video augmentation techniques including random temporal sampling, horizontal flipping, and photometric jitter could improve generalization. Finally, enabling end-to-end gradient flow through ResNet50 within the CNN-LSTM framework may allow it to develop motion-sensitive representations rather than relying solely on appearance features.

Citation
Shah, A.P., Lamare, J.B., Nguyen-Anh, T. and Hauptmann, A., 2018. CADP: A novel dataset for CCTV traffic camera based accident analysis. IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)
