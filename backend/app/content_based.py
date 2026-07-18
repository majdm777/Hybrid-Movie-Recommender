"""
Movie Recommender - Content-Based Model
-----------------------------------------
Builds a TF-IDF matrix from movies_full.csv's content_blob column,
computes cosine similarity between all movies, and exposes a function
to get the most similar movies to a given movieId.
contet
Run this AFTER fetch_tmdb.py (needs data/processed/movies_full.csv).
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "processed"


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
def load_movies():
    movies = pd.read_csv(DATA_DIR / "movies_full.csv")
    movies["content_blob"] = movies["content_blob"].fillna("")
    print(f"Loaded {len(movies)} movies")
    return movies


# ---------------------------------------------------------------------
# 2. BUILD TF-IDF MATRIX + SIMILARITY MATRIX
# ---------------------------------------------------------------------
def build_similarity_matrix(movies: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,   # cap vocabulary size, keeps memory/speed reasonable
        min_df=2,             # ignore terms that appear in only 1 movie (likely noise)
    )
    tfidf_matrix = vectorizer.fit_transform(movies["content_blob"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    return vectorizer, tfidf_matrix, similarity_matrix


# ---------------------------------------------------------------------
# 3. LOOKUP: movieId -> row index (and back)
# ---------------------------------------------------------------------
def build_index_maps(movies: pd.DataFrame):
    movieId_to_idx = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    idx_to_movieId = pd.Series(movies["movieId"].values, index=movies.index).to_dict()
    return movieId_to_idx, idx_to_movieId


# ---------------------------------------------------------------------
# 4. GET SIMILAR MOVIES
# ---------------------------------------------------------------------
def get_similar_movies(movie_id, movies, similarity_matrix, movieId_to_idx, idx_to_movieId, n=10):
    if movie_id not in movieId_to_idx:
        print(f"movieId {movie_id} not found.")
        return pd.DataFrame()

    idx = movieId_to_idx[movie_id]
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity, skip the movie itself (always similarity 1.0 with itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]

    result_rows = []
    for i, score in scores:
        mid = idx_to_movieId[i]
        title = movies.loc[movies["movieId"] == mid, "clean_title"].values[0]
        result_rows.append({"movieId": mid, "title": title, "similarity": round(score, 4)})

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------
# MAIN (quick manual test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    movies = load_movies()
    vectorizer, tfidf_matrix, similarity_matrix = build_similarity_matrix(movies)
    movieId_to_idx, idx_to_movieId = build_index_maps(movies)

    # Test: find movies similar to Toy Story (movieId 1)
    test_movie_id = 1
    test_title = movies.loc[movies["movieId"] == test_movie_id, "clean_title"].values[0]
    print(f"\nMovies similar to '{test_title}' (movieId={test_movie_id}):")

    recommendations = get_similar_movies(
        test_movie_id, movies, similarity_matrix, movieId_to_idx, idx_to_movieId, n=10
    )
    print(recommendations)
