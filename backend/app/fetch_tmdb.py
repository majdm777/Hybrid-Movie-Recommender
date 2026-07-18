"""
Movie Recommender - TMDB Fetch Script
--------------------------------------
Reads links.csv (movieId -> tmdbId), calls the TMDB API for each movie,
and caches the results locally as JSON files so re-runs don't re-hit
the API for movies already fetched.

Requires a backend/.env file with:
    TMDB_API_KEY=your_key_here

Run this AFTER prep_data.py from the backend directory.
"""

import os
import json
import time
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data" / "raw" / "ml-latest-small"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed"
CACHE_DIR = BACKEND_DIR / "data" / "tmdb_cache"
REQUEST_DELAY = 0.02           # seconds between calls, stay well under rate limits
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"

load_dotenv(BACKEND_DIR / ".env")
API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise RuntimeError("TMDB_API_KEY not found. Check your .env file.")

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 1. LOAD LINKS (movieId -> tmdbId)
# ---------------------------------------------------------------------
def load_links():
    links = pd.read_csv(DATA_DIR / "links.csv")
    # tmdbId can be missing (NaN) for some movies
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    print(f"Loaded {len(links)} movieId->tmdbId mappings")
    return links


# ---------------------------------------------------------------------
# 2. FETCH ONE MOVIE FROM TMDB (with local cache)
# ---------------------------------------------------------------------
def fetch_movie(tmdb_id):
    cache_path = CACHE_DIR / f"{tmdb_id}.json"

    # Return cached result if we already fetched this one
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{TMDB_BASE_URL}/{tmdb_id}"
    params = {
        "api_key": API_KEY,
        "append_to_response": "keywords",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            # Movie not found or other error - cache an empty marker so we don't retry forever
            data = {"error": resp.status_code}
        else:
            data = resp.json()
    except requests.RequestException as e:
        data = {"error": str(e)}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    time.sleep(REQUEST_DELAY)
    return data


# ---------------------------------------------------------------------
# 3. EXTRACT THE FIELDS WE CARE ABOUT
# ---------------------------------------------------------------------
def extract_fields(movie_id, tmdb_data):
    if not tmdb_data or "error" in tmdb_data:
        return {
            "movieId": movie_id,
            "overview": "",
            "keywords": "",
            "poster_path": "",
            "vote_average": None,
            "runtime": None,
        }

    keywords_list = tmdb_data.get("keywords", {}).get("keywords", [])
    keywords_text = " ".join(k["name"] for k in keywords_list)

    return {
        "movieId": movie_id,
        "overview": tmdb_data.get("overview") or "",
        "keywords": keywords_text,
        "poster_path": tmdb_data.get("poster_path") or "",
        "vote_average": tmdb_data.get("vote_average"),
        "runtime": tmdb_data.get("runtime"),
    }


# ---------------------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------------------
def fetch_all(links: pd.DataFrame):
    results = []
    total = len(links)

    for i, row in enumerate(links.itertuples(index=False), start=1):
        tmdb_data = fetch_movie(row.tmdbId)
        results.append(extract_fields(row.movieId, tmdb_data))

        if i % 250 == 0 or i == total:
            print(f"Fetched {i}/{total}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# 5. MERGE INTO movies_clean.csv
# ---------------------------------------------------------------------
def merge_and_save(tmdb_df: pd.DataFrame):
    movies_clean = pd.read_csv(OUTPUT_DIR / "movies_clean.csv")
    merged = movies_clean.merge(tmdb_df, on="movieId", how="left")

    # Build the combined text blob for content-based filtering
    merged["genres_clean"] = merged["genres_clean"].fillna("")
    merged["overview"] = merged["overview"].fillna("")
    merged["keywords"] = merged["keywords"].fillna("")

    merged["content_blob"] = (
        merged["genres_clean"] + " " + merged["overview"] + " " + merged["keywords"]
    )

    merged.to_csv(OUTPUT_DIR / "movies_full.csv", index=False)
    print(f"Saved movies_full.csv with shape {merged.shape}")
    return merged


if __name__ == "__main__":
    links = load_links()
    tmdb_df = fetch_all(links)
    merged = merge_and_save(tmdb_df)

    print("\nSample merged rows:")
    print(merged[["movieId", "clean_title", "overview", "keywords"]].head())
