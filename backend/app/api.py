"""
Movie Recommender - FastAPI Backend (v2: with real new-user support)
------------------------------------------------------------------------
Adds on top of the original hybrid model:
  - POST /users/signup           -> create a new user, get a unique userId
  - GET  /movies/to-rate         -> a batch of popular movies for a new user to rate
  - POST /ratings                -> submit a rating for a movie
  - GET  /recommend/{user_id}    -> now also works for NEW users (content-based,
                                     weighted by their own ratings only)

New users' ratings are stored in SQLite (db.py) and are completely separate
from the original MovieLens users - a new user only ever sees recommendations
based on THEIR OWN ratings, never anyone else's individual data.

Run with:
    uvicorn api:app --reload
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from .DataBase import db

DATA_DIR = "data/processed"
MIN_RATING = 0.5
MAX_RATING = 5.0
USER_RATING_THRESHOLD = 20      # ratings needed before leaning on CF-style signal
NEW_USER_MIN_RATINGS = 5        # minimum ratings before we personalize at all
MOVIE_RATING_THRESHOLD = 5

app = FastAPI(title="Hybrid Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {}


@app.on_event("startup")
def load_models():
    print("Loading data and training models...")

    movies = pd.read_csv(f"{DATA_DIR}/movies_full.csv")
    movies["content_blob"] = movies["content_blob"].fillna("")
    ratings_train = pd.read_csv(f"{DATA_DIR}/ratings_train.csv")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(movies["content_blob"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    movieId_to_idx = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    idx_to_movieId = pd.Series(movies["movieId"].values, index=movies.index).to_dict()

    reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))
    data = Dataset.load_from_df(ratings_train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    cf_model = SVD(n_factors=100, n_epochs=20, random_state=42)
    cf_model.fit(trainset)

    movie_counts = ratings_train["movieId"].value_counts()
    eligible_movie_ids = movie_counts[movie_counts >= MOVIE_RATING_THRESHOLD].index.tolist()

    state["movies"] = movies
    state["ratings_train"] = ratings_train
    state["similarity_matrix"] = similarity_matrix
    state["tfidf_matrix"] = tfidf_matrix
    state["movieId_to_idx"] = movieId_to_idx
    state["idx_to_movieId"] = idx_to_movieId
    state["cf_model"] = cf_model
    state["eligible_movie_ids"] = eligible_movie_ids

    # Set up the SQLite DB. New userIds start right after the highest
    # existing MovieLens userId, so there's no collision.
    max_existing_user_id = int(ratings_train["userId"].max())
    db.init_db(starting_user_id=max_existing_user_id + 1)

    print("Models ready.")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def movie_row_to_dict(row, score=None, source=None):
    d = {
        "movieId": int(row["movieId"]),
        "title": row["clean_title"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "poster_path": row.get("poster_path", "") or "",
    }
    if score is not None:
        d["score"] = round(float(score), 4)
    if source is not None:
        d["source"] = source
    return d


def get_similar_movies(movie_id, n=10):
    movies = state["movies"]
    similarity_matrix = state["similarity_matrix"]
    movieId_to_idx = state["movieId_to_idx"]
    idx_to_movieId = state["idx_to_movieId"]

    if movie_id not in movieId_to_idx:
        return []

    idx = movieId_to_idx[movie_id]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]

    results = []
    for i, score in scores:
        mid = idx_to_movieId[i]
        row = movies.loc[movies["movieId"] == mid].iloc[0]
        results.append(movie_row_to_dict(row, score=score, source="content"))
    return results


def get_cf_recommendations(user_id, n=10):
    movies = state["movies"]
    ratings_train = state["ratings_train"]
    cf_model = state["cf_model"]
    eligible_movie_ids = state["eligible_movie_ids"]

    already_rated = set(ratings_train.loc[ratings_train["userId"] == user_id, "movieId"])
    candidates = [mid for mid in eligible_movie_ids if mid not in already_rated]

    predictions = [(mid, cf_model.predict(user_id, mid).est) for mid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    results = []
    for mid, pred_rating in top_n:
        row = movies.loc[movies["movieId"] == mid].iloc[0]
        results.append(movie_row_to_dict(row, score=pred_rating, source="collaborative"))
    return results


def get_popular_movies(n=10):
    movies = state["movies"]
    ratings_train = state["ratings_train"]

    popular = (
        ratings_train.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .query("count >= @MOVIE_RATING_THRESHOLD")
        .sort_values("mean", ascending=False)
        .head(n)
    )

    results = []
    for mid, row in popular.iterrows():
        movie_row = movies.loc[movies["movieId"] == mid].iloc[0]
        results.append(movie_row_to_dict(movie_row, score=row["mean"], source="popularity"))
    return results


def get_content_recs_for_new_user(user_id, n=10):
    """
    For a brand-new (app) user: build a weighted profile vector from
    ONLY this user's own ratings (their rated movies' TF-IDF vectors,
    weighted by how highly they rated each one), then find the closest
    unrated movies to that profile. Never looks at any other user's data.
    """
    movies = state["movies"]
    tfidf_matrix = state["tfidf_matrix"]
    movieId_to_idx = state["movieId_to_idx"]

    user_ratings = db.get_ratings_for_user(user_id)
    if user_ratings.empty:
        return []

    rated_movie_ids = user_ratings["movie_id"].tolist()
    idxs = [movieId_to_idx[m] for m in rated_movie_ids if m in movieId_to_idx]
    weights = user_ratings.loc[user_ratings["movie_id"].isin(
        [m for m in rated_movie_ids if m in movieId_to_idx]
    ), "rating"].to_numpy()

    if not idxs:
        return []

    # Weighted average of the rated movies' TF-IDF vectors = user's "taste profile"
    rated_vectors = tfidf_matrix[idxs]
    weights_arr = np.array(weights).reshape(-1, 1)
    weighted = rated_vectors.multiply(weights_arr)
    profile_vector = np.asarray(weighted.mean(axis=0))

    similarities = cosine_similarity(profile_vector, tfidf_matrix).flatten()

    already_rated_set = set(rated_movie_ids)
    ranked_idx = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_idx:
        mid = movies.iloc[idx]["movieId"]
        if mid in already_rated_set:
            continue
        row = movies.iloc[idx]
        results.append(movie_row_to_dict(row, score=similarities[idx], source="content"))
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------
class SignupRequest(BaseModel):
    display_name: str 


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


# ---------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Hybrid Movie Recommender API is running."}


@app.post("/users/signup")
def signup(req: SignupRequest):
    new_id = db.create_new_user(display_name=req.display_name)
    return {"user_id": new_id}


@app.get("/movies/to-rate")
def movies_to_rate(n: int = Query(20, ge=5, le=100)):
    """
    Returns a batch of well-known (popular) movies for a new user to rate,
    so they have recognizable titles to react to.
    """
    return {"movies": get_popular_movies(n=n)}


@app.post("/ratings")
def submit_rating(req: RatingRequest):
    if req.rating < MIN_RATING or req.rating > MAX_RATING:
        raise HTTPException(status_code=400, detail=f"rating must be between {MIN_RATING} and {MAX_RATING}")
    if req.movie_id not in state["movieId_to_idx"]:
        raise HTTPException(status_code=404, detail="movieId not found")
    if not db.user_exists(req.user_id):
        raise HTTPException(status_code=404, detail="userId not found. Sign up first via /users/signup.")

    db.add_rating(req.user_id, req.movie_id, req.rating)
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    """
    Hybrid recommendation endpoint - handles BOTH:
      - original MovieLens users (userId found in ratings_train.csv)
      - new app users (userId assigned via /users/signup, ratings in SQLite)
    """
    ratings_train = state["ratings_train"]
    is_original_user = (ratings_train["userId"] == user_id).any()

    if is_original_user:
        user_rating_count = int((ratings_train["userId"] == user_id).sum())

        if user_rating_count < USER_RATING_THRESHOLD:
            strategy = "content"
            user_ratings = ratings_train[ratings_train["userId"] == user_id]
            seed_movie_id = int(user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"])
            recs = get_similar_movies(seed_movie_id, n=n)
        else:
            strategy = "collaborative"
            recs = get_cf_recommendations(user_id, n=n)

        return {
            "user_id": user_id,
            "user_rating_count": user_rating_count,
            "strategy": strategy,
            "recommendations": recs,
        }

    # --- New (app) user path ---
    if not db.user_exists(user_id):
        raise HTTPException(status_code=404, detail="userId not found. Sign up first via /users/signup.")

    user_ratings = db.get_ratings_for_user(user_id)
    rating_count = len(user_ratings)

    if rating_count < NEW_USER_MIN_RATINGS:
        return {
            "user_id": user_id,
            "user_rating_count": rating_count,
            "strategy": "popularity",
            "recommendations": get_popular_movies(n=n),
            "note": f"Rate at least {NEW_USER_MIN_RATINGS} movies to get personalized recommendations.",
        }

    strategy = "content_personalized"
    recs = get_content_recs_for_new_user(user_id, n=n)

    return {
        "user_id": user_id,
        "user_rating_count": rating_count,
        "strategy": strategy,
        "recommendations": recs,
    }


@app.get("/similar/{movie_id}")
def similar(movie_id: int, n: int = Query(10, ge=1, le=50)):
    if movie_id not in state["movieId_to_idx"]:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")
    recs = get_similar_movies(movie_id, n=n)
    return {"movie_id": movie_id, "recommendations": recs}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    movies = state["movies"]
    matches = movies[movies["clean_title"].str.contains(q, case=False, na=False)].head(limit)
    results = [movie_row_to_dict(row) for _, row in matches.iterrows()]
    return {"query": q, "results": results}