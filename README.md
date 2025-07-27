# Kindle-Review-

The process involves the following steps:

1.  **Data Loading and Initial Exploration**: The `all_kindle_review.csv` dataset is loaded into a pandas DataFrame, and initial checks are performed to understand the data structure and content.
2.  **Data Preparation**: Irrelevant columns are dropped, and the 'rating' column is converted into a binary classification (positive/negative). Text data in the 'reviewText' column is converted to lowercase.
3.  **Text Cleaning and Preprocessing**: Various text cleaning techniques are applied to the 'reviewText' column, including removing special characters, stopwords, URLs, HTML tags, and extra spaces. Lemmatization is also performed to reduce words to their base form.
4.  **Data Splitting**: The data is split into training and testing sets to prepare for model training and evaluation.
5.  **Text Vectorization**: The text data is transformed into numerical representations using both CountVectorizer (Bag-of-Words) and TfidfVectorizer.
6.  **Model Training**: Gaussian Naive Bayes models are trained on both the Bag-of-Words and TF-IDF vectorized data.
7.  **Model Evaluation**: The performance of the trained models is evaluated using confusion matrices and accuracy scores for both vectorization methods.
---
# Kindle Review Sentiment Analysis

This notebook demonstrates a sentiment analysis workflow applied to a dataset of Kindle reviews. The goal is to classify reviews as either positive or negative based on the review text.

## Dataset

The dataset used in this notebook is `all_kindle_review.csv`, which contains information about Kindle reviews, including the review text and rating.

## Workflow

The notebook follows a standard machine learning workflow for text classification:

1.  **Data Loading**: The dataset is loaded into a pandas DataFrame.
2.  **Data Inspection and Cleaning**:
    *   Initial inspection of the data is performed using `head()` and `info()`.
    *   Irrelevant columns are dropped.
    *   The 'rating' column, originally on a scale of 1-5, is converted into a binary label (0 for negative reviews, 1 for positive reviews). Reviews with a rating less than 3 are considered negative (0), and reviews with a rating of 3 or more are considered positive (1).
    *   The 'reviewText' column is converted to lowercase.
3.  **Text Preprocessing**:
    *   Various text cleaning techniques are applied to the 'reviewText' column to prepare it for vectorization. This includes removing special characters, stopwords, URLs, HTML tags, and extra spaces.
    *   Lemmatization is performed to reduce words to their base form, which helps in reducing the vocabulary size and improving model performance.
4.  **Data Splitting**: The preprocessed data is split into training and testing sets (80% for training, 20% for testing) to evaluate the model's performance on unseen data.
5.  **Text Vectorization**:
    *   **Bag-of-Words (BoW)**: The `CountVectorizer` is used to transform the text data into a matrix of token counts.
    *   **TF-IDF (Term Frequency-Inverse Document Frequency)**: The `TfidfVectorizer` is used to transform the text data into a matrix where each cell represents the importance of a word in a document relative to the entire corpus.
6.  **Model Training**:
    *   A Gaussian Naive Bayes model is trained separately on the Bag-of-Words and TF-IDF representations of the training data. Naive Bayes is a simple yet effective algorithm for text classification, particularly when dealing with high-dimensional data like text.
7.  **Model Evaluation**:
    *   The trained models are evaluated on the test set using standard classification metrics.
    *   **Confusion Matrix**: A confusion matrix is generated for each model to visualize the performance in terms of true positives, true negatives, false positives, and false negatives.
    *   **Accuracy Score**: The overall accuracy of each model is calculated and printed.

## Results

The notebook demonstrates the process of building and evaluating sentiment analysis models using two different text vectorization techniques. The accuracy scores for both the Bag-of-Words and TF-IDF models are calculated and displayed in the output of the respective cells.

The results show the accuracy of the Gaussian Naive Bayes model for both the Bag-of-Words and TF-IDF representations. Further analysis and potentially trying different models or hyperparameter tuning could improve the performance.

## How to Run the Notebook

1.  Upload the `all_kindle_review.csv` file to your Colab environment.
2.  Run all the cells in the notebook sequentially. The output of each cell will show the progress and results of each step.
