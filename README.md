# 💬 Natural Language Processing Projects

**Developed by** : Héctor M. Ramírez C.

**Datasets** :

* [Mental Health Text Dataset](https://www.kaggle.com/) 🧠 *(Kaggle)*
* [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) 🎬 *(Kaggle)*

---

## 📌 Overview

Natural Language Processing (NLP) has the power to unlock insights from massive amounts of unstructured text. This personal project explores **real-world NLP tasks** through mental health analysis and sentiment classification using open datasets.

From  **text preprocessing with spaCy** , to  **training models like Naive Bayes, Random Forest, SVM, and XGBoost** , this repository reflects a hands-on journey into NLP, with careful focus on model evaluation and meaningful problem framing.

---

## 🎯 Objective

The goal of this project is to demonstrate a clear understanding of  **NLP pipelines** , covering everything from raw text cleaning to deploying models capable of classifying mental health content or IMDB reviews.

What I aim to achieve:

* Build reproducible, scalable NLP pipelines with Python
* Compare traditional ML classifiers using TF-IDF and bag-of-ngrams
* Apply classification to emotionally and socially meaningful data
* Visualize results via confusion matrices and metrics (precision, recall, F1)

---

## 🧠 Project Highlights

* 📖  **Text Preprocessing** :
  * Tokenization, lemmatization with **spaCy**
  * Stopword removal, regex cleanup, vectorization
* 🤖  **Models Used** :
  * Naive Bayes
  * Random Forest
  * K-Nearest Neighbors
  * Support Vector Machine (SVM)
  * XGBoost
* 🧪  **NLP Tasks** :
  * Sentiment classification of IMDB reviews (positive/negative)
  * Detection of mental health conditions from user-generated text
* 🧰  **Tech Stack** :
  * Python, Scikit-learn, spaCy, XGBoost, Pandas, Matplotlib, Seaborn

---

## 🌱 Why This Matters to Me

I'm passionate about using  **AI and Machine Learning to address human and societal challenges** . Mental health, in particular, is an area where language plays a critical role in diagnosis and early detection. By exploring text data through these projects, I hope to contribute to meaningful use cases where machine learning can enhance well-being and empathy.

This work complements my academic background and career aspirations in machine learning and responsible AI development.

---

## 📁 Repository Structure

```
📦 NLP-Projects
├── 📂 data
│   ├── 📁 raw                   # 🧾 Original datasets (Mental Health, IMDB)
│   └── 📁 processed             # 🧹 Cleaned datasets ready for modeling
│
├── 📂 models
│   └── 📓 xgboost_mentalhealth_model.pkl # 🧠 ML on mental health with TF-IDF & n-grams
│
├── 📂 notebooks                 # 🧪 Additional exploration (optional)
│   ├── 📓 NLP_basics.ipynb                  # 🔤 Preprocessing walkthrough + IMDB classification
│   └── 📓 NLP_Sentiment_Mental_Health.ipynb # 🧠 ML on mental health with TF-IDF & n-grams
│
└── 📄 README.md                 # 📘 This file
```

---

## ✅ Next Steps

* [ ] Add deep learning models (LSTM, Transformer-based)
* [ ] Explore model explainability (e.g., SHAP for text)
* [ ] Deploy one of the models as a simple API or web app
