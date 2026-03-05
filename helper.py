import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """
    Cleans the storyline text by removing punctuation, stop words, 
    and unnecessary characters[cite: 24, 25].
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(filtered_tokens)

def get_recommendations(user_input, tfidf_vectorizer, tfidf_matrix, df, top_n=5):
    """
    Get movie recommendations based on user input using TF-IDF similarity.
    
    Args:
        user_input (str): User's movie preference description
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: Pre-computed TF-IDF matrix for all movies
        df (DataFrame): Movie dataset
        top_n (int): Number of recommendations to return (default: 5)
    
    Returns:
        DataFrame: Top N most similar movies with all their information
    """
    try:
        cleaned_input = preprocess_text(user_input)
        user_tfidf = tfidf_vectorizer.transform([cleaned_input])
        similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        
        # Get top N indices
        related_indices = similarity_scores.argsort()[-top_n:][::-1]
        print(df.iloc[related_indices])
        return df.iloc[related_indices]
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return None

