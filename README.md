# AI600 – Deep Learning Assignment 2
## Quick, Draw! Image Classification using MLP (No CNN)

---

## Student Information
- **Roll Number:** 25280074
- **Course:** AI600 – Deep Learning
- **Assignment:** Programming Assignment 2

---

# Project Overview

This project implements a MLP to classify hand-drawn doodles from a subset of the **Quick, Draw! dataset** containing 15 classes.

The goal was to:
- Compare shallow-wide vs deep-narrow architectures
- Analyze width vs depth trade-offs
- Design an efficient Champion model
- Generate predictions for a hidden test set leaderboard

---

# Repository Structure
DL_PA2/
│
├── processed_data/
│ ├── quickdraw_train.npz
│ ├── quickdraw_test.npz
│
├── models/
│ ├── pancake_best.pt
│ ├── tower_best.pt
│ ├── champion2_best.pt
│
├── submission.txt
├── 25280074__PA2.ipynb  ---> PART A, B , C, D, E(in PDF file)
├── README.md



---

# Environment Setup

## 1. Create Virtual Environment

python -m venv .venv

.\.venv\Scripts\activate

## 2. Install Required Packages
pip install torch torchvision numpy matplotlib pandas seaborn scikit-learn ipykernel

## 3. Register Jupyter Kernel
python -m ipykernel install --user --name dlpa2_env --display-name "DLPA2 (.venv)"


## How to Run the Projec
Open 25280074__PA2.ipynb

Select kernel: DLPA2 (.venv)

Run cells sequentially from top to bottom.

## Notebook Implementation Breakdown
Sections 1–3 (Provided)
Dataset loading
Normalization (0–1 scaling)
Train/validation split (80/20)
DataLoader setup

--- 

## Section 4 (Custom Implementation Added)

The following reusable framework was implemented:

Seed setting (reproducibility)

Parameter counter (3M constraint check)

Configurable MLP builder:

Variable depth

Variable width

Activation choice (ReLU, GELU, LeakyReLU)

Dropout

Batch Normalization

Training loop

Validation loop

Accuracy tracking

Model checkpoint saving

Loss/accuracy plotting

This framework was reused for Part A, B, and C experiments.

---

---

## Experiments (in 25280074_PA2.ipynb file)
### Part A – Pancake (Wide & Shallow)

Architecture:

784 → 2048 → 15

Parameters: 1,638,415

Best Validation Accuracy: 76.31%

Observation: Heavy overfitting

### Part B – Tower (Deep & Narrow)

Architecture:

784 → 256 → 256 → 256 → 256 → 256 → 256 → 15

Parameters: 536,847

Best Validation Accuracy: 77.94%

Observation: Better generalization with fewer parameters

### Part C – Champion (Final Model Used for Leaderboard)

Final Selected Architecture:

784 → 384 → 384 → 256 → 256 → 256 → 128 → 15

Activation: GELU

Batch Normalization: Yes

Dropout: 0.2

Weight Decay: 5e-4

Parameters: 717,583

Best Validation Accuracy: 78.69%

Best Epoch: 24

This model was selected for leaderboard submission due to strong performance and parameter efficiency.

---

---

## Generating Leaderboard Submission

After loading the best checkpoint:

model = final_model

Run the final inference section in the notebook.

This generates:

submission.txt

The contents of this file should be copied into the AI600 leaderboard portal.

---


---

## Part D – Analysis

The notebook includes:

Width vs Depth comparison

Confusion matrix visualization

Identification of top 2 confused class pairs:

basketball → soccerball

wheel → pizza

Discussion on data ambiguity vs model limitation

---
