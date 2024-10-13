import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

# Streamlit app configuration
st.set_page_config(page_title="Movie Recommendation System", page_icon=":movie_camera:", layout="wide")

# Cache the dataset loading
@st.cache_data
def load_data():
    movies = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/movies.csv')
    ratings = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/ratings.csv')
    tags = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/tags.csv')

    # Preprocess the data
    movies = movies.merge(
        tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index(),
        on='movieId',
        how='left'
    )
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['text'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['tag'].fillna('')

    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.columns = ['movieId', 'average_rating']
    movies = movies.merge(average_ratings, on='movieId', how='left')
    movies['rating'] = movies['average_rating'].fillna(0)

    return movies

# Load the SVD model
svd = joblib.load('svd_model.pkl')

# Load data
movies = load_data()

# Vectorization and dimensionality reduction
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(movies['text'])

# Transform the TF-IDF matrix to movie features using SVD
movie_features = svd.transform(tfidf_matrix)
movie_features = normalize(movie_features)

# Function to recommend movies
def recommend_movie(keyword, top_n=20):
    idx = movies[movies['text'].str.contains(keyword, case=False)].index.tolist()
    if not idx:
        return pd.DataFrame(columns=['title', 'genres', 'average_rating'])

    selected_movie_index = idx[0]
    selected_movie_vector = movie_features[selected_movie_index].reshape(1, -1)
    similarity_scores = np.dot(movie_features, selected_movie_vector.T).flatten()
    top_n_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = movies.iloc[top_n_indices]
    return recommendations[['title', 'genres', 'average_rating']]

# Streamlit app interface
st.title("🎬 Movie Recommendation System")
st.write("Find your next favorite movie by entering a keyword (movie name, genre):")

# User input
keyword = st.text_input("Keyword", placeholder="e.g., action, comedy, Titanic")

if st.button("Recommend"):
    recommendations = recommend_movie(keyword)

    if not recommendations.empty:
        st.write("### Recommendations:")
        # Create columns for the cards
        cols = st.columns(5)  # 5 cards per row
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with cols[i % 5]:  # Distribute cards in the columns
                st.markdown(
                    f"""
                    <div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin-bottom: 10px; height: 200px;'>
                        <h4 style='text-align: center;'>{row['title']}</h4>
                        <p style='text-align: center;'><strong>Genres:</strong> {row['genres']}</p>
                        <p style='text-align: center;'><strong>Average Rating:</strong> {row['average_rating']:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning(f"No movies found for keyword '{keyword}'")

# Footer
st.markdown("---")
st.write("Made By  ️Samit and Jaden 🎬")
