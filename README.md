# 📰 Fake News Detection using Machine Learning (NLP)

## 📘 Overview
This project predicts whether a news article is *real or fake* using *Natural Language Processing (NLP)* and *Machine Learning*.  
It uses text preprocessing and feature extraction (TF-IDF) to train a classification model that detects misleading or false information in online news.

---

## 📂 Dataset
- *Dataset name:* Fake News Dataset  
- *Source:* [Kaggle – Fake News Dataset](https://www.kaggle.com/c/fake-news/data)  
- *Columns:*
  - title — headline of the article  
  - text — content/body of the article  
  - label — 0 for Real news, 1 for Fake news  

> 📁 Save the dataset as fake_news.csv in your project directory.

---

## ⚙ Tech Stack
- *Language:* Python  
- *Libraries:*  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - NLTK  
  - Matplotlib  
  - Seaborn  

---

## 🧠 Model & Approach
1. *Data Loading* – Loaded dataset using Pandas.  
2. *Data Cleaning* – Removed missing values and unnecessary columns.  
3. *Text Preprocessing* –  
   - Removed punctuation and stopwords  
   - Converted text to lowercase  
   - Tokenized and lemmatized text using *NLTK*  
4. *Feature Extraction* – Used *TF-IDF Vectorizer* to convert text to numeric form.  
5. *Model Training* – Used *PassiveAggressiveClassifier* from Scikit-learn.  
6. *Evaluation* – Measured model accuracy and visualized confusion matrix.

---

## 📈 Results
| Metric | Score |
|---------|--------|
| *Accuracy* | 96.7% |
| *Precision* | 0.97 |
| *Recall* | 0.96 |

✅ *Best Model:* PassiveAggressiveClassifier  

---

## 🚀 How to Run the Project

1. *Clone this repository:*
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection

	2.	Install dependencies:

pip install -r requirements.txt


	3.	Run the script or notebook:

jupyter notebook fake_news_detection.ipynb

or

python fake_news_detection.py


	4.	Test with your own text:

sample = ["Aliens visited Earth last night!"]
sample_tfidf = vectorizer.transform(sample)
print(model.predict(sample_tfidf))

Output → 1 (Fake News) or 0 (Real News)

⸻

📊 Visualizations
	•	Confusion Matrix Heatmap
	•	Accuracy and Loss Curves
	•	Sample Predictions on Test Data

⸻

🌟 Key Learnings
	•	Learned text preprocessing with NLP
	•	Used TF-IDF for feature extraction
	•	Implemented and evaluated a binary text classification model
	•	Improved understanding of machine learning workflow and metrics

⸻

---

Would you like me to also generate the **requirements.txt** (so you can include it in your repo and easily install all libraries)?
