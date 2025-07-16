# Brain Response Classification

This project trains an image classifier using brain response topomaps to distinguish between reactions to **good** and **bad** designs. Each image is taken after 6 seconds of exposure and categorized as either positive (good) or negative (bad) based on brain activity.

## Dataset Structure

The dataset contains brain topomap images organized in the following structure:

## Dataset Structure

The dataset contains brain topomap images organized in the following structure:

```text
/topomaps
├── good
│   ├── Good_6s_1.png
│   ├── Good_6s_2.png
│   └── ...
└── bad
    ├── Bad_6s_1.png
    ├── Bad_6s_2.png
    └── ...
```

- Label `1` is assigned to `good` responses.
- Label `0` is assigned to `bad` responses.
- Sparse label encoding is used.

## Files Included

- `train.py` — Training script using PyTorch and ResNet-18.
- `eval.py` — Evaluation script that loads a trained model and predicts on new data.
- `model.pth` — Trained model checkpoint file.
- `requirements.txt` — Dependencies used (optional).
- `test_eval.ipynb` — For testing the working of eval.py.

## Training Details

- Model: ResNet-18 (with modified final layer for binary classification).
- Loss Function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Data Augmentations:
  - Resize to 224x224
  - Random horizontal flip
  - Random rotation (±10°)
- Class imbalance is handled using a `WeightedRandomSampler`.
- Dataset is split: 80% training / 20% validation.
- Training runs for 20 epochs by default.

To train the model:
  ```text
python train.py
```

This will save model.pth in the project directory.

## Evaluation

The eval.py script loads the trained model and runs predictions on a given directory of new topomaps:
```text
from eval import load_and_predict

predictions = load_and_predict("path/to/new/topomaps", "model.pth")
print(predictions)  # outputs a list of 0s and 1s
```
All images are resized to 224x224 and passed through the trained model. Predictions are thresholded using sigmoid > 0.5.

## Requirements
The following packages are needed to run the code:

- torch==2.6.0+cu124
- torchvision==0.21.0+cu124
- numpy==1.26.4
- Pillow==10.4.0
- scikit-learn==1.6.1
