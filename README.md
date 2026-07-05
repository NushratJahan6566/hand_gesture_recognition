**Hand Gesture Recognition using MediaPipe + MLP**

Small project to recognize hand gestures in real-time using a webcam. Instead of training a CNN on raw images, I used MediaPipe to get hand landmarks first, then trained a simple MLP on those. Worked well even with a small dataset.

### Why landmarks, not raw images

Tried MobileNetV2 on raw images first. Only ~374 images, so it overfit badly. Got just 70% accuracy. Switched to landmarks (21 keypoints, x/y/z each = 63 numbers per image). Much smaller feature space, worked way better. MLP hit 93% test accuracy.

### Dataset

374 images, 4 gestures. Collected myself, tried to vary lighting, background, hand angle, and used both hands.

| Class | Images |
|---|---|
| Open Palm | 94 |
| Peace | 96 |
| Point Up | 90 |
| Thumbs Up | 94 |

Dataset not included here (too many files). Can share if needed, just open an issue.

### How it works

1. Resize and clean up images
2. Run MediaPipe Hands, get 21 landmarks
3. Flatten to 63-dim vector, normalize
4. Feed into MLP (Dense + ReLU + BatchNorm + Dropout)
5. Train with Adam, early stopping, LR reduction on plateau

### Results

| Model | Test Accuracy |
|---|---|
| MLP | 93.1% |
| SVM | 84.5% |
| Logistic Regression | 87.9% |

MLP won. Probably handles the non-linear stuff in landmark positions better.

Confusion matrix and full report are in the notebook. Gets a bit confused between two similar-looking gestures, otherwise solid.

### Files

- `mediapipe.ipynb` - full pipeline
- `best_mlp_model.h5` - trained model
- `scaler.pkl` - scaler for inference
- `landmark.png` - sample landmark visualization

### What doesn't work great

- Only 4 static gestures for now
- Struggles in low light or weird hand angles
- More data would help a lot, 374 images isn't much

### Next steps

Maybe add more gesture classes. Curious how it holds up in a different lighting setup.
