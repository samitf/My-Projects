import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import cupy as cp
import joblib

movies = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/movies.csv')
ratings = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/ratings.csv')
tags = pd.read_csv('C:/Users/samit/Downloads/archive/ml-25m/tags.csv')

movies = movies.merge(
    tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna().astype(str))).reset_index(),
    on='movieId',
    how='left'
)

movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['text'] = movies['title'] + ' ' + movies['genres'] + ' ' + movies['tag'].fillna('')

print("Movies DataFrame shape:", movies.shape)
print("First 5 rows of movies DataFrame:\n", movies.head())
print("Missing values in each column:\n", movies.isnull().sum())

genre_counts = movies['genres'].str.split(expand=True).stack().value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index[:10], y=genre_counts.values[:10])
plt.title('Top 10 Genres')
plt.xlabel('Genres')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(ratings['rating'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(np.arange(0.5, 5.5, 0.5))
plt.grid(axis='y')
plt.show()

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

predicted_ratings = np.array([train_data['rating'].mean()] * len(test_data))

mae = mean_absolute_error(test_data['rating'], predicted_ratings)
rmse = np.sqrt(mean_squared_error(test_data['rating'], predicted_ratings))
threshold = 1
correct_predictions = np.sum(np.abs(test_data['rating'] - predicted_ratings) <= threshold)
accuracy = correct_predictions / len(test_data)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Accuracy: {accuracy:.4f}")

sizes = [len(train_data), len(test_data)]
labels = ['Training Set', 'Testing Set']
plt.figure(figsize=(8, 5))
plt.bar(labels, sizes, color=['skyblue', 'salmon'])
plt.title('Training and Testing Set Size')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(movies['text'])
tfidf_gpu = cp.asarray(tfidf_matrix.todense())

svd = TruncatedSVD(n_components=100, algorithm='randomized')
svd.fit(tfidf_gpu.get())
movie_features = svd.transform(tfidf_gpu.get())
movie_features = normalize(movie_features)

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(svd, 'svd_model.pkl')

average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.columns = ['movieId', 'average_rating']
movies = movies.merge(average_ratings, on='movieId', how='left')
movies['rating'] = movies['average_rating'].fillna(0)

def recommend_movie(keyword, top_n=20):
    idx = movies[movies['text'].str.contains(keyword, case=False)].index.tolist()
    if not idx:
        print(f"No movies found for keyword '{keyword}'")
        return []

    selected_movie_index = idx[0]
    selected_movie_vector = movie_features[selected_movie_index].reshape(1, -1)
    similarity_scores = np.dot(movie_features, selected_movie_vector.T).flatten()
    top_n_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = movies.iloc[top_n_indices]

    true_labels, pred_labels = [], []
    for _, row in recommendations.iterrows():
        true_rating = row['average_rating']
        if true_rating > 0:
            true_labels.append(1 if true_rating >= 3 else 0)
            pred_labels.append(1)
        else:
            print(f"Skipping movie {row['title']} with average rating {true_rating:.2f}")

    if true_labels:
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, zero_division=0)
        rec = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        print(f"\nRecommendations for '{keyword}':")
        for _, row in recommendations.iterrows():
            print(f"Title: {row['title']} | Genres: {row['genres']} | Average Rating: {row['average_rating']:.2f} | Movie ID: {row['movieId']}")

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1 Score: {f1:.2f}")
    else:
        print("No valid true labels to calculate metrics.")

    return recommendations

keyword = input("Enter a keyword (movie name, actor, director, genre): ")
recommend_movie(keyword)
