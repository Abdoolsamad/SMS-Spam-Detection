
# SMS Spam Detection Using Machine Learning

An intelligent SMS classification system built with Python and Scikit-learn to identify spam and legitimate text messages using Natural Language Processing (NLP) and machine learning.

The project includes a Streamlit web application that allows users to enter SMS messages and receive real-time predictions.

## Project Overview

SMS spam messages are commonly used for fraudulent promotions, unwanted advertising, and phishing attempts.

This project explores how machine learning can automatically distinguish spam messages from legitimate messages.

The system uses TF-IDF for text feature extraction and compares multiple classification algorithms to identify the best-performing model.

## Features

- SMS spam and legitimate message classification
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Comparison of four machine learning algorithms
- Class balancing to improve spam detection
- Real-time predictions through a Streamlit interface
- Estimated class probabilities
- Saved model and vectorizer for reuse

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Dataset

The project uses the UCI SMS Spam Collection dataset.

Dataset source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The original dataset contains 5,572 SMS messages.

After removing 403 duplicate records, 5,169 messages remained for model development.

## Methodology

### 1. Data Exploration

- Inspected dataset structure and dimensions.
- Examined spam and legitimate message distribution.
- Checked missing values and duplicate records.

### 2. Text Preprocessing

- Converted messages to lowercase.
- Removed unnecessary whitespace.
- Removed duplicate records.

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text messages into numerical features.

The vectorizer was fitted on the training data only to prevent data leakage.

### 4. Model Training

Four machine learning algorithms were implemented and evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Balanced Logistic Regression
4. Complement Naive Bayes

An 80/20 stratified train-test split was used.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Model Performance

The following results were obtained on the held-out test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 97.38% | 100.00% | 78.91% | 88.21% |
| Logistic Regression | 96.22% | 98.90% | 70.31% | 82.19% |
| Balanced Logistic Regression | 97.97% | 90.84% | 92.97% | 91.89% |
| Complement Naive Bayes | 94.96% | 72.89% | 94.53% | 82.31% |

### Best Performing Model

Balanced Logistic Regression achieved the best overall F1-score among the four tested configurations.

- Accuracy: 97.97%
- Spam Precision: 90.84%
- Spam Recall: 92.97%
- Spam F1-score: 91.89%

Class balancing improved the model's ability to identify spam messages, although it introduced some false positive predictions.

## Application Interface

The Streamlit application provides:

- A text input area for SMS messages.
- Example spam and legitimate messages.
- A button to analyze messages.
- Classification results.
- Estimated spam and ham probabilities.

## Project Structure

```text
SMS-Spam-Detection/
│
├── app.py
├── SMS_Spam_Detection.ipynb
├── best_sms_spam_model.pkl
├── best_tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SMS-Spam-Detection.git
```

### 2. Navigate into the project folder

```bash
cd SMS-Spam-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Future Improvements

- Test additional machine learning algorithms.
- Tune classification thresholds.
- Evaluate performance using cross-validation.
- Explore more advanced NLP techniques.
- Deploy the application online.

## Author

Abdulsamad Bashir

BSc Information Technology  
Bayero University Kano, Nigeria

GitHub: https://github.com/Abdoolsamad

## Acknowledgements

- UCI Machine Learning Repository for providing the SMS Spam Collection dataset.
- Scikit-learn documentation.
- Streamlit documentation.
