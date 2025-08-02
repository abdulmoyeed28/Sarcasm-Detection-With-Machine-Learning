
# 🧠 Sarcasm Detection with Machine Learning & NLP

This project uses Machine Learning and NLP to detect sarcasm in news headlines. By analyzing text patterns and contextual cues, it classifies headlines as sarcastic or not. It combines data exploration, BERT-based modeling, and a web interface for real-time prediction.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## 🔍 Overview

This end-to-end NLP project applies both traditional ML techniques and state-of-the-art transformer models (like BERT) to perform sarcasm detection on news headlines. From data preprocessing and tokenization to model training and deployment, it offers a comprehensive view of real-world sentence classification.

---

## 💭 Problem Statement

Sarcasm is a complex and subtle form of expression that often relies on context, tone, and contradiction. In NLP, detecting sarcasm is essential for improving sentiment analysis, chatbot understanding, and content moderation. This project answers:
- Can we automatically detect sarcasm in news headlines?
- How effective are modern ML and DL techniques in this task?

---

## 📂 Dataset

- **Source**: [Kaggle – Sarcasm News Headlines Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Fields**:
  - `is_sarcastic`: 1 = sarcastic, 0 = non-sarcastic
  - `headline`: News headline text
  - `article_link`: Link to the original article

---

## ⚙️ Installation

Make sure you have Python 3.7+ installed.

```bash
git clone https://github.com/abdulmoyeed28/Sarcasm-Detection-With-Machine-Learning.git
cd Sarcasm-Detection-With-Machine-Learning
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

- **Tokenizer**: Pretrained BERT tokenizer
- **Model**: BERT base model with classification head
- **Library**: HuggingFace Transformers + PyTorch
- **Metrics**: Accuracy, F1-Score, Confusion Matrix

---

## 📊 Results

- **Accuracy**: ~87%
- **F1 Score**: ~86.4%
- Confusion matrix and loss curves are available in the notebook.
