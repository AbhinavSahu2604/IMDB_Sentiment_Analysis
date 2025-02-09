# IMDB_Sentiment_Analysis
Python | NLTK | Spacy | Wordcloud | TensorFlow | Keras

This project focuses on building a robust **Natural Language Processing (NLP)** pipeline for sentiment analysis on IMDB movie reviews. The aim is to classify reviews as positive or negative using machine learning models and deep learning architectures.  

---

## 📌 **Project Overview**
- **Objective**: To analyze IMDB movie reviews and predict their sentiment.
- **Dataset**: The IMDB Dataset containing 50,000 reviews labeled as either positive or negative.
- **Key Features**:
  - Data cleaning and preprocessing (removing noise, tokenization, and lemmatization).
  - Exploratory Data Analysis (EDA) using word clouds, n-grams, and visualizations.
  - Feature engineering using TF-IDF and Count Vectorizers.
  - Building classification models using traditional machine learning and deep learning.

---

## 🛠️ **Tech Stack**
- **Programming Language**: Python
- **Libraries**:
  - **NLP**: NLTK, SpaCy, WordCloud, Tokenizers
  - **Visualization**: Matplotlib, Seaborn
  - **Machine Learning**: Scikit-learn, XGBoost
  - **Deep Learning**: TensorFlow, Keras
- **Development Tools**: Jupyter Notebook

---

## 🚀 **Project Workflow**
1. **Data Preprocessing**
   - Removed special characters, stop words, and performed tokenization.
   - Used TF-IDF and Count Vectorizer for feature extraction.
2. **Exploratory Data Analysis (EDA)**
   - Generated word clouds for positive and negative reviews.
   - Analyzed frequent unigrams, bigrams, and trigrams using bar plots.
3. **Model Building**
   - Traditional ML Models:
     - Logistic Regression, SVM, Naive Bayes, XGBoost.
   - Deep Learning Models:
     - LSTM (Long Short-Term Memory) for sequential data analysis.
   - Hyperparameter tuning for model optimization.
4. **Evaluation**
   - Metrics used: Accuracy, Precision, Recall, F1-Score, and ROC-AUC Curve.
   - Compared ML and DL models to identify the best-performing approach.

---

## 📊 **Model Evaluation and Results**
### Performance Metrics:
- **Accuracy**: **88.20%**
- **Loss**: **0.3068**

### Confusion Matrix
The confusion matrix below illustrates the model's performance:
- **True Positives (TP)**: 4424
- **True Negatives (TN)**: 4320
- **False Positives (FP)**: 619
- **False Negatives (FN)**: 554


### Inference:
- The model demonstrates strong performance in classifying reviews with relatively low false positives and negatives.
- Effective use of LSTM layers to capture sequential patterns in text data contributed to high accuracy.

---

## 🔍 **Highlights**
- **Visualization**:
  - Word clouds showcasing common words in positive and negative reviews.
  - Accuracy and loss plots for training and validation during model evaluation.
- **Feature Engineering**:
  - Implementation of TF-IDF and Count Vectorization for text representation.
  - Effective handling of imbalanced datasets.



