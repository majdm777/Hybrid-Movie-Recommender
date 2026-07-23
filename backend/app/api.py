"""
Movie Recommender - FastAPI Backend
--------------------------------------
Loads the data and trains the content-based + collaborative filtering
models ONCE at startup, keeps them in memory, and exposes endpoints
for the frontend to call.

Run with:
    python -m uvicorn app.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

from pathlib import Path
import math

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "processed"
MIN_RATING = 0.5
MAX_RATING = 5.0
USER_RATING_THRESHOLD = 20
MOVIE_RATING_THRESHOLD = 5


app = FastAPI(title="Hybrid Movie Recommender API")

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's URL before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Globals populated once at startup (see @app.on_event("startup"))
# ---------------------------------------------------------------------
state = {}


@app.on_event("startup")
def load_models():
    print("Loading data and training models... this runs once at startup.")

    movies = pd.read_csv(DATA_DIR / "movies_full.csv")
    movies["content_blob"] = movies["content_blob"].fillna("")
    ratings_train = pd.read_csv(DATA_DIR / "ratings_train.csv")

    # --- Content-based ---
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(movies["content_blob"])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    movieId_to_idx = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    idx_to_movieId = pd.Series(movies["movieId"].values, index=movies.index).to_dict()

    # --- Collaborative filtering ---
    reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))
    data = Dataset.load_from_df(ratings_train[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    cf_model = SVD(n_factors=100, n_epochs=20, random_state=42)
    cf_model.fit(trainset)

    # --- Movies eligible for CF serving ---
    movie_counts = ratings_train["movieId"].value_counts()
    eligible_movie_ids = movie_counts[movie_counts >= MOVIE_RATING_THRESHOLD].index.tolist()

    state["movies"] = movies
    state["ratings_train"] = ratings_train
    state["similarity_matrix"] = similarity_matrix
    state["movieId_to_idx"] = movieId_to_idx
    state["idx_to_movieId"] = idx_to_movieId
    state["cf_model"] = cf_model
    state["eligible_movie_ids"] = eligible_movie_ids

    print("Models ready.")


# ---------------------------------------------------------------------
# Helper functions (same logic as hybrid.py)
# ---------------------------------------------------------------------
def get_similar_movies(movie_id, min_rate=0.2):
    movies = state["movies"]
    similarity_matrix = state["similarity_matrix"]
    movieId_to_idx = state["movieId_to_idx"]
    idx_to_movieId = state["idx_to_movieId"]

    if movie_id not in movieId_to_idx:
        return []

    idx = movieId_to_idx[movie_id]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:]


    results = []
    for i, score in scores:

        if score < min_rate:
            break
        else:    
            mid = idx_to_movieId[i]
            row = movies.loc[movies["movieId"] == mid].iloc[0]

            poster = row.get("poster_path", "")
            if pd.isna(poster):
                poster =""
    
    
            results.append({
                "movieId": int(mid),
                "title": row["clean_title"],
                "year": None if pd.isna(row["year"]) else int(row["year"]),
                "poster_path": poster,
                "score": round(float(score), 4),
                "source": "content",
            })

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
        results.append({
            "movieId": int(mid),
            "title": row["clean_title"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "poster_path": row.get("poster_path", ""),
            "score": round(float(pred_rating), 3),
            "source": "collaborative",
        })
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
        results.append({
            "movieId": int(mid),
            "title": movie_row["clean_title"],
            "year": None if pd.isna(movie_row["year"]) else int(movie_row["year"]),
            "poster_path": movie_row.get("poster_path", ""),
            "score": round(float(row["mean"]), 3),
            "source": "popularity",
        })
    return results


# ---------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Hybrid Movie Recommender API is running."}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    """
    Hybrid recommendation endpoint. Switches strategy based on how many
    ratings this user has in the training data.
    """
    ratings_train = state["ratings_train"]
    user_rating_count = int((ratings_train["userId"] == user_id).sum())

    if user_rating_count == 0:
        strategy = "popularity"
        recs = get_popular_movies(n=n)
    elif user_rating_count < USER_RATING_THRESHOLD:
        strategy = "content"
        user_ratings = ratings_train[ratings_train["userId"] == user_id]
        seed_movie_id = int(user_ratings.sort_values("rating", ascending=False).iloc[0]["movieId"])
        recs = get_similar_movies(seed_movie_id, MIN_RATING)
    else:
        strategy = "collaborative"
        recs = get_cf_recommendations(user_id, n=n)

    return {
        "user_id": user_id,
        "user_rating_count": user_rating_count,
        "strategy": strategy,
        "recommendations": recs,
    }


@app.get("/similar/{movie_id}")
def similar(movie_id: int, n: float = Query(0.25, ge=0.01, le=1)):
    """
    Content-based "movies like this" endpoint - independent of any user.
    """
    if movie_id not in state["movieId_to_idx"]:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")

    recs = get_similar_movies(movie_id, min_rate=n)
    return {"movie_id": movie_id, "recommendations": recs}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    """
    Simple title search, for the frontend's search bar (e.g. to pick a
    seed movie for the /similar endpoint).
    """
    movies = state["movies"]
    matches = movies[movies["clean_title"].str.contains(q, case=False, na=False)].head(limit)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "movieId": int(row["movieId"]),
            "title": row["clean_title"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "poster_path": row.get("poster_path", ""),
        })
    return {"query": q, "results": results}

# @app.get("/movies/surprise")


