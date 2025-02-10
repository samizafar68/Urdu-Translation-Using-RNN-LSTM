# Natural Language Processing

## English-to-Urdu Translation using Many-to-Many RNN and LSTM

This repository contains the implementation of **Many-to-Many Recurrent Neural Network (RNN)** and **Long Short-Term Memory (LSTM)** models for English-to-Urdu language translation. The project investigates the limitations of RNNs in language translation and demonstrates how LSTMs can effectively address these challenges.

## Project Overview

The goal of this project is to implement and compare **RNN** and **LSTM** models for English-to-Urdu language translation. This project is divided into three parts:

1. **Many-to-Many RNN Implementation**: Building and training an RNN model for translation.
2. **Limitations of RNNs**: Discussing the challenges faced by RNNs in language translation.
3. **LSTM Implementation**: Replacing RNN layers with LSTM layers to overcome RNN limitations.

---

## Dataset

The dataset used for this project is the **parallel-corpus.xlsx**, which contains parallel sentences in English and Urdu. The dataset is split into training, validation, and test sets for model training and evaluation.

---

## Technologies and Tools

- **Languages**: Python
- **Deep Learning Libraries**: PyTorch
- **Text Processing Libraries**: NLTK, spaCy
- **Evaluation Metrics**: BLEU Score, Accuracy
- **Modeling Techniques**:
  - **Many-to-Many RNN**: A basic recurrent neural network designed to handle sequence-to-sequence translation tasks.
  - **LSTM (Long Short-Term Memory)**: A type of RNN used to solve the vanishing gradient problem, suitable for learning long-term dependencies in sequences.

---

## Tasks

### Part 1: Many-to-Many RNN Implementation

1. **Data Preparation**:
   - Download the **parallel-corpus.xlsx** dataset.
   - Preprocess the English and Urdu text by tokenizing and cleaning the data.
   - Split the dataset into training, validation, and test sets.

2. **Model Architecture**:
   - Build an RNN-based many-to-many architecture using TensorFlow or PyTorch.
   - Train the model on the English-to-Urdu dataset.

3. **Evaluation**:
   - Evaluate the model using BLEU score and accuracy.
   - Tested Some example translations from the test set to show the model's performance.

### Part 2: Reporting the Limitations of RNNs

1. **Limitations**:
   - **Exploding/Vanishing Gradients**: RNNs struggle to learn long sequences due to vanishing/exploding gradient problems.
   - **Difficulty in Capturing Long-Term Dependencies**: RNNs struggle with long sentences, particularly with languages like Urdu, which have complex grammatical structures.
   - **Performance on Large Datasets**: RNNs may not generalize well to large and diverse datasets.
   - Tested some examples where the RNN model struggles with these issues.

### Part 3: Resolving RNN Limitations Using LSTM

1. **LSTM Implementation**:
   - Modify the RNN-based model by replacing the RNN layers with LSTM layers.
   
2. **Comparison**:
   - Compare the performance of the RNN and LSTM models using BLEU score and accuracy.
   - Discuss how LSTM addresses the limitations of RNNs.

3. **Final**:
   - Summarize the performance comparison.
   - Discuss the improvements made by using LSTM over RNN.
   - Highlight any remaining challenges in English-to-Urdu translation with LSTM and suggest further improvements.

---

## Evaluation Metrics

- **BLEU Score**: Measures the quality of the translation by comparing it to reference translations. Higher BLEU scores indicate better translation quality.
- **Accuracy**: Measures the percentage of correctly translated words or sentences. It is used to evaluate how well the model performs at a high level.

---
