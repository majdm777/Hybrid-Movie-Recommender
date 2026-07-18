"""
Movie Recommender - Hybrid Switching Logic
---------------------------------------------
Combines the content-based model and the collaborative filtering (SVD)
model using a switching rule:

  - Users with FEW ratings (cold-start)  -> content-based recommendations
  - Users with ENOUGH ratings            -> collaborative filtering
  - Movies with FEW ratings are excluded from being surfaced via CF
    (kept eligible via content-based only)

Run this AFTER content_based.py and collaborative_filtering.py logic
are confirmed working (this script reuses their functions).
"""

from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "processed"
MIN_RATING = 0.5
MAX_RATING = 5.0

# --- Thresholds decided during planning ---
USER_RATING_THRESHOLD = 20   # below this -> treat as cold-start, use content-based
MOVIE_RATING_THRESHOLD = 5   # movies with fewer ratings than this are excluded from CF serving


# ---------------------------------------------------------------------
# 1. LOAD EVERYTHING NEEDED
# ---------------------------------------------------------------------
def load_all():
    movies = pd.read_csv(DATA_DIR / "movies_full.csv")
    movies["content_blob"] = movies["content_blob"].fillna("")

    ratings_train = pd.read_csv(DATA_DIR / "ratings_train.csv")

    return movies, ratings_train


# ---------------------------------------------------------------------
# 2. BUILD CONTENT-BASED COMPONENTS
# ---------------------------------------------------------------------
def build_content_model(movies: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(movies["content_blob"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    movieId_to_idx = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    idx_to_movieId = pd.Series(movies["movieId"].values, index=movies.index).to_dict()

    return similarity_matrix, movieId_to_idx, idx_to_movieId


def get_similar_movies(movie_id, movies, similarity_matrix, movieId_to_idx, idx_to_movieId, n=10):
    if movie_id not in movieId_to_idx:
        return pd.DataFrame()

    idx = movieId_to_idx[movie_id]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]

    rows = []
    for i, score in scores:
        mid = idx_to_movieId[i]
        title = movies.loc[movies["movieId"] == mid, "clean_title"].values[0]
        rows.append({"movieId": mid, "title": title, "score": round(score, 4), "source": "content"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. BUILD CF COMPONENT
# ---------------------------------------------------------------------
def build_cf_model(ratings_train: pd.DataFrame):
    reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))
    data = Dataset.load_from_df(ratings_train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)
    return model


def get_cf_recommendations(user_id, model, movies, ratings_train, eligible_movie_ids, n=10):
    already_rated = set(ratings_train.loc[ratings_train["userId"] == user_id, "movieId"])
    candidates = [mid for mid in eligible_movie_ids if mid not in already_rated]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    rows = []
    for mid, pred_rating in top_n:
        title = movies.loc[movies["movieId"] == mid, "clean_title"].values[0]
        rows.append({"movieId": mid, "title": title, "score": round(pred_rating, 3), "source": "collaborative"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. HYBRID SWITCHING LOGIC
# ---------------------------------------------------------------------
def recommend_for_user(
    user_id,
    movies,
    ratings_train,
    cf_model,
    similarity_matrix,
    movieId_to_idx,
    idx_to_movieId,
    n=10,
):
    """
    Main entry point. Decides which strategy to use based on how many
    ratings this user has, and returns a DataFrame of recommendations.
    """
    user_rating_count = (ratings_train["userId"] == user_id).sum()

    # Movies eligible for CF-based serving (enough ratings to trust the signal)
    movie_counts = ratings_train["movieId"].value_counts()
    eligible_movie_ids = movie_counts[movie_counts >= MOVIE_RATING_THRESHOLD].index.tolist()

    if user_rating_count == 0:
        # True cold-start: no ratings at all -> can't personalize yet.
        # Fall back to most-rated / highest-average as a placeholder strategy.
        print(f"User {user_id} has no ratings. Cold-start: returning popular movies.")
        popular = (
            ratings_train.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .query("count >= @MOVIE_RATING_THRESHOLD")
            .sort_values("mean", ascending=False)
            .head(n)
        )
        rows = []
        for mid, row in popular.iterrows():
            title = movies.loc[movies["movieId"] == mid, "clean_title"].values[0]
            rows.append({"movieId": mid, "title": title, "score": round(row["mean"], 3), "source": "popularity"})
        return pd.DataFrame(rows)

    elif user_rating_count < USER_RATING_THRESHOLD:
        # Low history -> lean on content-based, seeded from their highest-rated movie
        print(f"User {user_id} has {user_rating_count} ratings (< {USER_RATING_THRESHOLD}). Using content-based.")
        user_ratings = ratings_train[ratings_train["userId"] == user_id]
        seed_movie_id = user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"]

        return get_similar_movies(
            seed_movie_id, movies, similarity_matrix, movieId_to_idx, idx_to_movieId, n=n
        )

    else:
        # Enough history -> collaborative filtering
        print(f"User {user_id} has {user_rating_count} ratings (>= {USER_RATING_THRESHOLD}). Using collaborative filtering.")
        return get_cf_recommendations(
            user_id, cf_model, movies, ratings_train, eligible_movie_ids, n=n
        )


# ---------------------------------------------------------------------
# MAIN (manual test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    movies, ratings_train = load_all()

    print("Building content-based model...")
    similarity_matrix, movieId_to_idx, idx_to_movieId = build_content_model(movies)

    print("Training collaborative filtering model...")
    cf_model = build_cf_model(ratings_train)

    # Test with a real user from the dataset (should trigger CF, since min ratings = 20)
    test_user_id = 1
    print(f"\n--- Recommendations for userId={test_user_id} ---")
    recs = recommend_for_user(
        test_user_id, movies, ratings_train, cf_model, similarity_matrix, movieId_to_idx, idx_to_movieId, n=10
    )
    print(recs)

    # Test with a fake new user (userId not in the dataset -> 0 ratings -> cold-start path)
    fake_user_id = 999999
    print(f"\n--- Recommendations for userId={fake_user_id} (cold-start test) ---")
    recs_cold = recommend_for_user(
        fake_user_id, movies, ratings_train, cf_model, similarity_matrix, movieId_to_idx, idx_to_movieId, n=10
    )
    print(recs_cold)
