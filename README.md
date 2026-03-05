# 🎬 IMDB Movie Recommendation System

An intelligent movie recommendation system built with **Streamlit** that uses **TF-IDF vectorization** and **cosine similarity** to suggest movies based on user preferences.

## ✨ Features

- **User-Friendly Web Interface**: Built with Streamlit for an intuitive and responsive UI
- **Smart Recommendations**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to analyze movie storylines
- **Fast Performance**: Leverages pre-computed TF-IDF models for instant recommendations
- **Top 5 Results**: Displays the 5 most relevant movies matching user preferences
- **Flexible Search**: Enter any movie description, genre, or keyword to find recommendations
- **Detailed Results**: Shows movie titles and complete storylines for each recommendation

## 📋 Project Structure

```
IMBD Movie Recommendation System/
├── app.py                          # Main Streamlit application
├── helper.py                       # Helper functions for text processing and recommendations
├── pyproject.toml                  # Python project configuration
├── README.md                       # Project documentation
|
├── Data/                           # Movie dataset
│   ├── imdb_movies_2024.csv        # Raw IMDB dataset
│   └── imdb_movies_2024_cleaned.csv # Cleaned dataset
|
├── Models/                         # Pre-trained ML models
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│   └── tfidf_matrix.pkl            # Pre-computed TF-IDF matrix
|
└── NoteBook/                       # Jupyter notebooks for data processing
    ├── Data Extration.ipynb
    ├── Data Processing.ipynb
    └── Prediction.ipynb
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd "d:\IMBD Movie Recommendation System"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually install:
   ```bash
   pip install streamlit pandas scikit-learn nltk
   ```

## 🎯 How to Run

1. **Activate the virtual environment** (if not already active)
   ```bash
   .venv\Scripts\activate
   ```

2. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

## 💡 How It Works

### Recommendation Algorithm

The system uses **TF-IDF Vectorization** and **Cosine Similarity**:

1. **Text Preprocessing**
   - Converts text to lowercase
   - Removes punctuation and special characters
   - Tokenizes into words
   - Removes English stopwords (common words like "the", "a", "and")

2. **TF-IDF Vectorization**
   - Converts processed text into numerical vectors
   - Measures word importance across the movie dataset
   - Pre-computed and stored for fast retrieval

3. **Cosine Similarity Matching**
   - Compares user input with all movie storylines
   - Calculates similarity scores (0-1 range)
   - Returns the 5 movies with highest similarity scores

### Data Flow
```
User Input → Text Preprocessing → TF-IDF Vectorization 
→ Cosine Similarity Calculation → Top 5 Recommendations
```

## 📝 Usage Examples

### Example 1: Action Movie
**Input**: "superhero action movie"
**Output**: Movies with superhero and action-packed storylines

### Example 2: Romance
**Input**: "romantic love story"
**Output**: Movies with romantic themes and love stories

### Example 3: Thriller
**Input**: "mystery thriller with twist"
**Output**: Mystery and thriller movies with unexpected plot turns

### Example 4: Comedy
**Input**: "funny comedy movie"
**Output**: Comedy films with humorous content

## 📊 Dataset

- **Source**: IMDB Movies 2024
- **Size**: 1000+ movies
- **Columns Used**: Title, Storyline
- **Format**: CSV

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Library** | Scikit-learn |
| **NLP** | NLTK |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Pickle |

## 📚 Key Components

### `app.py`
Main Streamlit application containing:
- Page configuration and styling
- Data and model loading with caching
- UI components (text input, search button)
- Recommendation display

### `helper.py`
Helper functions including:
- `preprocess_text()`: Cleans and tokenizes movie descriptions
- `get_recommendations()`: Calculates similarity and returns top 5 matches

## ⚙️ Functions

### `preprocess_text(text)`
Cleans input text by:
- Converting to lowercase
- Removing punctuation
- Tokenizing into words
- Removing stopwords

### `get_recommendations(user_input, tfidf_vectorizer, tfidf_matrix, df, top_n=5)`
Returns top N movie recommendations based on user input
- **Parameters**: User query, TF-IDF vectorizer, TF-IDF matrix, movie dataframe
- **Returns**: DataFrame with top N recommendations

## 🎨 UI Features

- **Header**: Eye-catching title with Netflix-inspired styling
- **Sidebar**: Dataset information
- **Input Section**: Single-line search with input field and button
- **Results**: Ranked list with movie titles and complete storylines
- **Loading Indicator**: Shows progress while searching

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Ensure `Data/imdb_movies_2024.csv` exists |
| "Model file not found" | Check `Models/` directory contains pickle files |
| Import errors | Run `pip install -r requirements.txt` |
| Port already in use | Use `streamlit run app.py --server.port 8502` |

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
nltk>=3.8.0
numpy>=1.24.0
```

## 🔧 Configuration

The app uses Streamlit's caching for performance:
- `@st.cache_data`: Caches dataset loading
- `@st.cache_resource`: Caches ML model loading

No additional configuration needed - works out of the box!

## 🚀 Future Enhancements

- Add filters by genre, year, rating
- Implement collaborative filtering
- Add user history and personalized recommendations
- Include movie posters and ratings
- Multi-language support

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

IMDB Movie Recommendation System - 2024

## 💬 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all files are in the correct directories
3. Ensure Python environment is properly activated
4. Check that all dependencies are installed

---

**Enjoy discovering your next favorite movie! 🍿**
