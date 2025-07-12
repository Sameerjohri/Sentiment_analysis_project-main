
📝 Sentiment Analysis of Product Reviews using Naive Bayes
This project applies Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier to perform sentiment analysis on product reviews. It classifies reviews as Positive, Negative, or Neutral based on the review text. A web app built with Streamlit allows users to interactively input a review and receive real-time predictions.

📁 Project Structure

 
Sentiment_analysis_project/
├── app.py                   # Streamlit web app
├── train_and_export.py      # Training script for data cleaning, model building & exporting
├── amazon_reviews.csv       # Dataset of product reviews
├── naive_bayes_model.pkl    # Trained Naive Bayes model
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
🔍 Dataset Overview
Source: Amazon product reviews dataset (can be extended to Flipkart, Yelp, etc.)

File: amazon_reviews.csv

Fields Used:

reviewText: Raw customer review text

overall: Rating (1–5) mapped to sentiment:

1–2 → Negative

3 → Neutral

4–5 → Positive

⚙️ Workflow
1. Data Preprocessing
Text normalization (lowercasing, punctuation removal)

Stopword removal (using NLTK)

Stemming using PorterStemmer

2. Feature Engineering
TF-IDF vectorization with 3000 most important features

3. Model Building
Algorithm: Multinomial Naive Bayes

Train/Test Split: 80/20

Model exported using joblib

4. Evaluation Metrics
Accuracy, Precision, Recall

Confusion Matrix visualization

5. Deployment
Streamlit web application for real-time sentiment prediction


⚙️ Setup Instructions

1. Clone the Repository
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app

2. Create Virtual Environment (optional)
   python -m venv .venv
   .venv\Scripts\activate   (Windows)
   source .venv/bin/activate  (Mac/Linux)

3. Install Dependencies
   pip install -r requirements.txt

▶️ Run Locally
   streamlit run app.py

☁️ Deploy to Streamlit Cloud
1. Push this folder to GitHub
2. Go to https://streamlit.io/cloud
3. Click “Create App”
4. Select your repo, branch, and app.py
5. Deploy and get your public link

📄 Sample Input
"This product is ver good! I’m really satisfied."


Author
Sameer Johri
🎓 B.Tech CSE | 💡 Data Science Enthusiast
📧 Email: [sameerjohri8@gmail.com]
🔗 GitHub: https://github.com/Sameerjohri
🔗 LinkedIn: https://linkedin.com/in/sameer-johri

📃 License
This project is open-source and available under the MIT License.
