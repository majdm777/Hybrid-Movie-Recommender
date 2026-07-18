"""
Movie Recommender - Collaborative Filtering Model (SVD)
----------------------------------------------------------
Trains an SVD matrix factorization model on ratings_train.csv
using the surprise library, evaluates it on ratings_test.csv,
and exposes a function to get top-N recommendations for a user.

Run this AFTER prep_data.py (needs data/processed/ratings_train.csv
and data/processed/ratings_test.csv).
"""

from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "processed"
MIN_RATING = 0.5
MAX_RATING = 5.0


# ---------------------------------------------------------------------
# 1. LOAD DATA INTO SURPRISE FORMAT
# ---------------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(DATA_DIR / "ratings_train.csv")
    test_df = pd.read_csv(DATA_DIR / "ratings_test.csv")

    reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))

    # surprise needs exactly these 3 columns, in this order
    train_data = Dataset.load_from_df(
        train_df[["userId", "movieId", "rating"]], reader
    )
    trainset = train_data.build_full_trainset()

    # Build testset as a list of (user, item, true_rating) tuples
    testset = list(
        test_df[["userId", "movieId", "rating"]].itertuples(index=False, name=None)
    )

    print(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")
    return trainset, testset


# ---------------------------------------------------------------------
# 2. TRAIN SVD MODEL
# ---------------------------------------------------------------------
def train_model(trainset):
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)
    print("SVD model trained.")
    return model


# ---------------------------------------------------------------------
# 3. EVALUATE ON TEST SET
# ---------------------------------------------------------------------
def evaluate_model(model, testset):
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    return rmse, mae


# ---------------------------------------------------------------------
# 4. GET TOP-N RECOMMENDATIONS FOR A USER
# ---------------------------------------------------------------------
def get_cf_recommendations(user_id, model, movies_df, ratings_train_df, n=10):
    # Movies the user has already rated - exclude these from recommendations
    already_rated = set(
        ratings_train_df.loc[ratings_train_df["userId"] == user_id, "movieId"]
    )

    all_movie_ids = movies_df["movieId"].unique()
    candidates = [mid for mid in all_movie_ids if mid not in already_rated]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    result_rows = []
    for mid, pred_rating in top_n:
        title = movies_df.loc[movies_df["movieId"] == mid, "clean_title"].values[0]
        result_rows.append({"movieId": mid, "title": title, "predicted_rating": round(pred_rating, 3)})

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------
# MAIN (train, evaluate, quick manual test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    trainset, testset = load_data()
    model = train_model(trainset)

    print("\nEvaluation on test set:")
    evaluate_model(model, testset)

    # Quick manual test: recommend for userId=1
    movies_df = pd.read_csv(DATA_DIR / "movies_full.csv")
    ratings_train_df = pd.read_csv(DATA_DIR / "ratings_train.csv")

    test_user_id = 1
    print(f"\nTop 10 recommendations for userId={test_user_id}:")
    recommendations = get_cf_recommendations(test_user_id, model, movies_df, ratings_train_df, n=20)
    print(recommendations)
