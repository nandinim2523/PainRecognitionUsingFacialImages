# PainRecognitionUsingFacialImages
Pain recognition system using facial images with deep learning (CNN) and traditional ML (HOG, LBP) techniques on the BP4D Spontaneous dataset.
---------
```
## Project Motivation
Pain detection is a key task in affective computing with potential applications in healthcare, especially for non-verbal patients. This project explores how **Convolutional Neural Network (CNNs)** and **feature-based methods (HOG and LBP)** compare in classifying pain from facial images.

---------

## Dataset

- **Name**: BP4D Spontaneous
- **Structure**:\

Affective_Computing/Pain_classification/
├── training/
│   ├── pain/
│   └── no_pain/
├── validation/
│   ├── pain/
│   └── no_pain/
└── testing/
    ├── pain/
    └── no_pain/

```

- Each folder contains facial image frames labeled as either "Pain" or "no pain".
--------

## Technologies Used
- **Programming Language:** Python
- **Libraries & Frameworls:** PyTorch, OpenCV, scikit-learn, torchvision, matplotlib, seaborn

-----
## Dataset and Preprocessing

- **Dataset:** BP4D Spontaneous Facial Expression Database
- **Data Preparation:**
  - All Images were resized to 224x224 pixels and converted to RGB format.
  - Training images underwent augmentation techniques such as random horizontal flips, slight rotations (±15°), and color jitter (brightness and contrast adjustments).
  - Images were normalized and transformed into tensors.
- **Data Split:** The dataset was divided into training, validaion, and testing sets with balanced class distributions.
- **Labels:** Binary classification task distinguisging between pain and no-pain expressions.


-----

## Models and Methods

### 1. Deep Learning 
- **Convolutional Neural Netwrok (CNN):** A custome-desinged CNN architecture was trained from scratch.
- **ResNet18:** Transfer learning was employed using a pretrained ResNet18 model fine-tuned on our dataset.

### 2. Traditional Machine Learning
- **Feature Extraction Technique:**
  - Local Binary Patterns (LBP)
  - Histogram of Oriented Gradients (HOG)
  - Scale-Invariant Feature Transform (SIFT)
- **Classifiers:**
  - Support Vectore Machine (SVM)
  - Random Forest (RF)


------

## Evalution Strategy

- **Validation:** Stratifies splits ensured balanced representation of classes during training and validation.
- **Testing:** Final performance was evaluated on a seperate, unseen test set.
- **Metrics:** Accuracy, precision, Recall, F1-score, Confusion Matrix , and Precision-Recall curves (for deep models).

-------

## Results Summary

| Model                | Validation Accuracy | Test Accuracy | Precision | Recall | F1-score |
|----------------------|---------------------|---------------|-----------|--------|----------|
| CNN                  | ~64%                | 51%           | 89%       | 44%    | 45%      |
| ResNet18             | 55%                 | 42%           | 86%       | 42%    | 41%      |
| LBP + Random Forest   | 49%                 | 31%           | 37%       | 31%    | 26%      |
| HOG + Random Forest   | 100%                | 37%           | 32%       | 51%    | 33%      |
| HOG + SVM            | 100%                | 43%           | 58%       | 43%    | 40%      |
| SIFT + SVM           | 52%                 | 49%           | 49%       | 49%    | 47%      |
| SIFT + Random Forest  | 46%                 | 40%           | 57%       | 53%    | 36%      |

*Note: Some traditional feature-based models achieved high validation accuracy but performed variably on the test set.*

------

## Challenges Encountered

- **Class Imbalance:** The dataset contained more no-pain than pain examples, impacting recall rates, especially in deep learning methods.
- **Computational Constraints:** Training deep models was time-consuming without GPU acceleration.
- **Feature Extraction Costs:** SIFT and HOG computations were resource-intensive, affecting runtime.
- **Hybrid Model Complexity:** Due to time limits, integration of handcrafted and deep features was not realized.


-----

## How to Run

1. clone the repo
2. install dependencies:
   pip install -r requiremnts.txt
3. Run training and evaluation scripts as needed
   - Train CNN
   - Extract traditional features and train classifiers
