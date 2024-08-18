import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
@st.cache_resource
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
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

# Function to display classification report
def display_classification_report(y_test, predictions, model_name):
    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"### {model_name} Classification Report")
    st.dataframe(report_df)

# Main function to create the Streamlit app
def main():
    st.title("IMDB Movie Reviews Sentiment Analysis Dashboard")
    st.write("This application performs sentiment analysis on IMDB movie reviews and provides insights through visualizations.")

    df = load_data()
    vectorizer, nb_model, lr_model, X_test_vec, y_test, nb_predictions, lr_predictions = train_models(df)

    st.subheader("Model Performance Overview")

    # Display model performance metrics
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Naive Bayes Model")
        st.write(f"Accuracy: {accuracy_score(y_test, nb_predictions):.2f}")
        plot_confusion_matrix(y_test, nb_predictions, "Naive Bayes")
        display_classification_report(y_test, nb_predictions, "Naive Bayes")

    with col2:
        st.write("### Logistic Regression Model")
        st.write(f"Accuracy: {accuracy_score(y_test, lr_predictions):.2f}")
        plot_confusion_matrix(y_test, lr_predictions, "Logistic Regression")
        display_classification_report(y_test, lr_predictions, "Logistic Regression")

    st.subheader("Predict Sentiment of Your Own Review")
    user_input = st.text_area("Enter a movie review:")
    if st.button("Predict"):
        user_input_preprocessed = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_preprocessed])
        nb_prediction = nb_model.predict(user_input_vectorized)
        lr_prediction = lr_model.predict(user_input_vectorized)
        st.write(f"Naive Bayes Prediction: {'Positive' if nb_prediction[0] == 1 else 'Negative'}")
        st.write(f"Logistic Regression Prediction: {'Positive' if lr_prediction[0] == 1 else 'Negative'}")

if __name__ == '__main__':
    main()
