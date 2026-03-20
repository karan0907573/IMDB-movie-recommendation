import streamlit as st
import pandas as pd
import pickle
from helper import preprocess_text, get_recommendations

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .recommendation-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .title-main {
        font-size: 2.5em;
        color: #e50914;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the movie dataset"""
    try:
        df = pd.read_csv("Data/imdb_movies_2024.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Data/imdb_movies_2024_cleaned.csv' exists.")
        return None

@st.cache_resource
def load_models():
    """Load pre-trained TF-IDF models"""
    try:
        with open("Models/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        
        with open("Models/tfidf_matrix.pkl", "rb") as f:
            tfidf_matrix = pickle.load(f)

        return tfidf_vectorizer, tfidf_matrix
    except FileNotFoundError as e:
        st.error(f"Model file not found: {str(e)}")
        return None, None

def display_recommendations(recommendations):
    """Display recommendations in a nice format with optional visualization"""
    if recommendations is None or len(recommendations) == 0:
        st.warning("No recommendations found. Please try a different search.")
        return
    
    st.subheader("🎯 Top 5 Recommendations")
    
    # Toggle between card view and chart view
    view_mode = st.toggle("📊 Show Visualization", value=False)
    
    if view_mode:
        # Bar chart visualization of match scores
        chart_data = recommendations[['Title', 'Match Score (%)']].copy()
        chart_data = chart_data.set_index('Title').sort_values('Match Score (%)')
        st.bar_chart(chart_data, horizontal=True)
    else:
        for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
            with st.container():
                col1, col2, col3 = st.columns([0.1, 0.75, 0.15])
                
                with col1:
                    st.markdown(f"### #{idx}")
                
                with col2:
                    st.markdown(f"### {movie.get('Title', 'N/A')}")
                
                with col3:
                    st.metric("Match", f"{movie['Match Score (%)']}%")
                
                if 'Storyline' in movie and pd.notna(movie['Storyline']):
                    st.write(movie['Storyline'])
                
                st.divider()

def main():
    """Main application"""
    # Header
    st.markdown('<div class="title-main">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
    st.write("Find your next favorite movie based on your preferences!")
    
    # Load data and models
    df = load_data()
    tfidf_vectorizer, tfidf_matrix = load_models()
    
    if df is None or tfidf_vectorizer is None:
        st.error("Failed to load required data and models. Please check the files.")
        return
    
    # Sidebar information
    with st.sidebar:
        st.metric("Total Movies", len(df))
    
    # Main input section
    st.divider()
    
    search_col, button_col = st.columns([0.85, 0.15])
    
    with search_col:
        user_input = st.text_input(
            "🔍 What kind of movie are you looking for?",
            placeholder="e.g., 'superhero action movie', 'romantic comedy', 'thriller with twist'",
            help="Describe the movie genre, theme, or any keywords you're interested in.",
            label_visibility="collapsed"
        )
    
    with button_col:
        search_button = st.button("Search", use_container_width=True, type="primary")
    
    st.divider()
    
    # Get recommendations
    if search_button or user_input:
        if not user_input.strip():
            st.warning("Please enter a movie description to get recommendations.")
        else:
            with st.spinner("Finding the perfect movies for you..."):
                recommendations = get_recommendations(
                    user_input, 
                    tfidf_vectorizer, 
                    tfidf_matrix, 
                    df
                )
                
                if recommendations is not None and len(recommendations) > 0:
                    display_recommendations(recommendations)
                else:
                    st.warning("No recommendations found. Try different keywords!")

if __name__ == "__main__":
    main()
