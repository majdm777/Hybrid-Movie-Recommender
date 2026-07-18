"""
Movie Recommender - Data Prep Script
------------------------------------
Reads ratings.csv and movies.csv from ml-latest-small, cleans them,
and saves ready-to-use files for the content-based and collaborative
filtering modules.

Run this module from the backend directory. Source data is read from
data/raw/ml-latest-small and processed files are written to data/processed.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data" / "raw" / "ml-latest-small"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed"
TEST_SIZE = 0.2         # 20% of ratings go to test set
RANDOM_STATE = 42       # for reproducibility

# ---------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------
def load_data():
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"

    missing_files = [path.name for path in (ratings_path, movies_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required data file(s) in {DATA_DIR}: {', '.join(missing_files)}"
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    validate_inputs(ratings, movies)
    print(f"Loaded ratings: {ratings.shape}, movies: {movies.shape}")
    return ratings, movies


def validate_inputs(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """Fail early when the source files are not valid MovieLens tables."""
    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}

    missing_ratings = required_ratings - set(ratings.columns)
    missing_movies = required_movies - set(movies.columns)
    if missing_ratings or missing_movies:
        details = []
        if missing_ratings:
            details.append(f"ratings.csv: {', '.join(sorted(missing_ratings))}")
        if missing_movies:
            details.append(f"movies.csv: {', '.join(sorted(missing_movies))}")
        raise ValueError(f"Missing required column(s): {'; '.join(details)}")

    if ratings.empty or movies.empty:
        raise ValueError("ratings.csv and movies.csv must both contain data.")
    if ratings[list(required_ratings)].isna().any().any():
        raise ValueError("ratings.csv contains missing values in required columns.")
    if movies["movieId"].isna().any() or movies["title"].isna().any() or movies["genres"].isna().any():
        raise ValueError("movies.csv contains missing values in required columns.")
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values.")


# ---------------------------------------------------------------------
# 2. CLEAN MOVIES: extract year, clean title, parse genres
# ---------------------------------------------------------------------
def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()

    # Extract year like "(1995)" at the end of the title
    year_extracted = movies["title"].str.extract(r"\((\d{4})\)\s*$")
    movies["year"] = pd.to_numeric(year_extracted[0], errors="coerce")

    # Clean title: strip the trailing "(YYYY)"
    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    )

    # Genre parsing: "(no genres listed)" -> empty string, else pipe -> space
    def parse_genres(g):
        if g == "(no genres listed)":
            return ""
        return g.replace("|", " ")

    movies["genres_clean"] = movies["genres"].apply(parse_genres)

    return movies


# ---------------------------------------------------------------------
# 3. SPLIT RATINGS: random 80/20
# ---------------------------------------------------------------------
def split_ratings(ratings: pd.DataFrame):
    train, test = train_test_split(
        ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


# ---------------------------------------------------------------------
# 4. SAVE
# ---------------------------------------------------------------------
def save_outputs(movies, train, test):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(OUTPUT_DIR / "movies_clean.csv", index=False)
    train.to_csv(OUTPUT_DIR / "ratings_train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "ratings_test.csv", index=False)
    print("Saved: movies_clean.csv, ratings_train.csv, ratings_test.csv")


if __name__ == "__main__":
    ratings, movies = load_data()
    movies_clean = clean_movies(movies)
    train, test = split_ratings(ratings)
    save_outputs(movies_clean, train, test)

    # Quick sanity check output
    print("\nSample cleaned movies:")
    print(movies_clean[["movieId", "clean_title", "year", "genres_clean"]].head())
