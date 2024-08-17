import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load dataset function
@st.cache_data
def load_data():
    df = pd.read_csv('IMDB Dataset 2.csv')
    df['review'] = df['review'].apply(preprocess_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

# Train models function
@st.cache_data
def train_models(df):
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_predictions = nb_model.predict(X_test_vec)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    lr_predictions = lr_model.predict(X_test_vec)

    return vectorizer, nb_model, lr_model, X_test_vec, y_test, nb_predictions, lr_predictions

# Function to display confusion matrix
def plot_confusion_matrix(y_test, predictions, title):
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    st.pyplot(fig)

# Main function to create the Streamlit dashboard
def main():
    st.set_page_config(page_title="IMDB Movie Reviews Dashboard", layout="wide")

    st.sidebar.title("IMDB Sentiment Analysis")
    st.sidebar.write("This dashboard allows you to analyze sentiment in IMDB movie reviews using different models.")

    df = load_data()
    vectorizer, nb_model, lr_model, X_test_vec, y_test, nb_predictions, lr_predictions = train_models(df)

    st.title("IMDB Movie Reviews Sentiment Analysis Dashboard")

    # Layout: Model Performance
    st.subheader("Model Performance Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Naive Bayes Model")
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        st.write(f"**Accuracy:** {nb_accuracy:.2f}")
        st.write(classification_report(y_test, nb_predictions))
        plot_confusion_matrix(y_test, nb_predictions, "Naive Bayes Confusion Matrix")

    with col2:
        st.write("### Logistic Regression Model")
        lr_accuracy = accuracy_score(y_test, lr_predictions)
        st.write(f"**Accuracy:** {lr_accuracy:.2f}")
        st.write(classification_report(y_test, lr_predictions))
        plot_confusion_matrix(y_test, lr_predictions, "Logistic Regression Confusion Matrix")

    # Layout: Model Comparison
    st.subheader("Model Accuracy Comparison")
    model_accuracies = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'Accuracy': [nb_accuracy, lr_accuracy]
    })
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=model_accuracies, ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Layout: User Input
    st.subheader("Predict Sentiment of Your Own Review")
    user_input = st.text_area("Enter a movie review:")
    if st.button("Predict"):
        user_input_preprocessed = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_preprocessed])
        nb_prediction = nb_model.predict(user_input_vectorized)
        lr_prediction = lr_model.predict(user_input_vectorized)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Naive Bayes Prediction")
            st.write(f"**Prediction:** {'Positive' if nb_prediction[0] == 1 else 'Negative'}")

        with col2:
            st.write("### Logistic Regression Prediction")
            st.write(f"**Prediction:** {'Positive' if lr_prediction[0] == 1 else 'Negative'}")

if __name__ == '__main__':
    main()
