# Real vs Fake Face Image Classification

A deep learning project that classifies face images as **real** or **AI-generated (fake)**.
Four models are built, tuned, and compared: two custom CNNs trained from scratch and two
transfer-learning models (ResNet50 and EfficientNetB0) pretrained on ImageNet.

---

## Dataset

**RVF10K** from Kaggle: [sachchitkunichetty/rvf10k](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k)

| Split      | Real  | Fake  | Total  |
|------------|-------|-------|--------|
| Train      | 3,500 | 3,500 | 7,000  |
| Validation | 1,500 | 1,500 | 3,000  |

The dataset is perfectly balanced, so accuracy is a reliable headline metric.

---

## Pipeline

1. **Exploratory Data Analysis**: class distribution and sample visualization.
2. **Preprocessing and Data Augmentation** (training set only): horizontal flip, zoom, rotation,
   width/height shift, and brightness adjustment. Each training image gets 2 augmented copies,
   giving **21,000** training images.
3. **Model building and hyperparameter tuning**, using a different tuning strategy for each model.
4. **Evaluation**: Accuracy, Precision, Recall, F1-score, AUC-ROC, confusion matrix, and
   training curves.

---

## Models

| Model | Architecture | Input | Tuning Method |
|-------|-------------|-------|---------------|
| **Custom CNN V1** | 4 Conv2D blocks + BatchNorm + SpatialDropout + L2 (~259K params) | 128×128 | Grid Search (12 combinations) |
| **Custom CNN V3** | 5 SeparableConv2D blocks with progressive SpatialDropout, label smoothing, gradient clipping, warmup + cosine LR schedule (~47K params) | 128×128 | Hyperband (Keras Tuner) |
| **ResNet50** | ImageNet-pretrained backbone + custom dense head | 224×224 | Random Search (6 combinations) |
| **EfficientNetB0** | ImageNet-pretrained backbone + custom dense head | 224×224 | Random Search (6 combinations) |

**Overfitting-aware model selection:** for each tuning run, the gap between training and
validation accuracy was tracked. For the transfer-learning models, only configurations with a
gap below 0.08 were kept, and the one with the highest validation accuracy was chosen.

---

## Results (Validation Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Custom CNN V1** | **0.833** | 0.839 | **0.825** | **0.832** | **0.911** |
| ResNet50 | 0.818 | 0.816 | 0.822 | 0.819 | 0.897 |
| EfficientNetB0 | 0.773 | 0.776 | 0.767 | 0.771 | 0.854 |
| Custom CNN V3 | 0.770 | **0.863** | 0.642 | 0.736 | 0.868 |

*Precision and Recall are reported with "Real" as the positive class.*

### Key Findings

- **A simple CNN trained from scratch beat both pretrained models.** ImageNet features
  are general-purpose and do not capture the subtle artifacts that separate real faces from
  generated ones.
- **Custom CNN V3** reached high precision but low recall on real faces: it tends to label real
  faces as fake.

---

## How to Run

1. Open the notebook in **Google Colab**. A GPU runtime is recommended.
2. Run the first cells to download the dataset with `kagglehub`. You will need Kaggle access.
3. The notebook mounts Google Drive to cache the dataset and the augmented images. Update the
   paths if your Drive structure is different.
4. Run the cells in order: EDA → Augmentation → Models → Evaluation.

## Tech Stack

Python · TensorFlow / Keras · Keras Tuner · scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · Google Colab

## Repository Structure

```
├── Real_vs_Fake_project.ipynb   # Full notebook: EDA, augmentation, models, tuning, evaluation
└── README.md
```
