# 📰 Fake News Detection using Machine Learning

## 📌 Overview

This project is a **Fake News Detection System** built using Python and
Machine Learning.\
It classifies news articles as **Fake** or **Real** based on their
textual content.

The model uses: - TF-IDF Vectorization - Support Vector Machine
(LinearSVC)

------------------------------------------------------------------------

## 🚀 Features

-   Combines news title and content for better accuracy
-   Cleans and preprocesses text data
-   Uses n-grams (unigrams + bigrams)
-   Trains a machine learning model
-   Evaluates performance using accuracy and classification report
-   Allows user input for real-time prediction

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Regular Expressions (re)

------------------------------------------------------------------------

## 📂 Dataset

This project uses two datasets: - `True.csv` → Real news - `Fake.csv` →
Fake news

⚠️ Update file paths in the code before running.

------------------------------------------------------------------------

## ⚙️ Installation

1.  Clone the repository:

``` bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

2.  Install dependencies:

``` bash
pip install pandas numpy scikit-learn
```

------------------------------------------------------------------------

## ▶️ Usage

Run the script:

``` bash
python main.py
```

Then enter any news text when prompted:

    Enter news text:

The model will predict: - Fake News - Real News

------------------------------------------------------------------------

## 🧠 How It Works

1.  Loads datasets
2.  Labels fake and real news
3.  Cleans text data:
    -   Lowercase conversion
    -   Removes URLs
    -   Removes special characters
4.  Combines title + text
5.  Converts text into numerical features using TF-IDF
6.  Trains a Linear Support Vector Classifier
7.  Evaluates model performance
8.  Predicts new input

------------------------------------------------------------------------

## 📊 Model Evaluation

-   Accuracy Score
-   Classification Report:
    -   Precision
    -   Recall
    -   F1-score

------------------------------------------------------------------------

## 📌 Example Output

    Accuracy: 0.98

    Classification Report:
                  precision    recall  f1-score   support
    fake          0.98       0.99      0.98
    real          0.99       0.98      0.98

------------------------------------------------------------------------

## ⚠️ Notes

-   Make sure dataset paths are correct
-   Larger datasets improve accuracy
-   Model can be improved using deep learning (LSTM, BERT)

------------------------------------------------------------------------

## 📈 Future Improvements

-   Add GUI (Tkinter / Web App)
-   Deploy using Flask or Streamlit
-   Use advanced NLP models (BERT, Transformers)
-   Add dataset auto-download

------------------------------------------------------------------------

## 👨‍💻 Author

Roshan Jangir

------------------------------------------------------------------------

## 📄 License

This project is open-source and available under the MIT License.
