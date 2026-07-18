# Hybrid Movie Recommendation System — Project Overview

## 1. Project Goal
Build a movie recommendation system that combines **content-based filtering** and **collaborative filtering** into a **hybrid** model, served through an API and a simple frontend.

---

## 2. Approach: Hybrid Recommendation

Two components feed into the final recommendation:

- **Content-based filtering**: recommends movies similar to ones a user liked, based on features like genres, overview text, and keywords. Works even for new/cold-start users or rarely-rated movies.
- **Collaborative filtering (CF)**: recommends based on patterns across many users' ratings ("people who liked X also liked Y"), using matrix factorization (SVD).

### Hybrid strategy: Switching Hybrid
- **Cold-start users / sparsely-rated movies** → lean on content-based recommendations
- **Users with rating history / well-rated movies** → lean on collaborative filtering
- This is more defensible than arbitrary blending — it explicitly solves the cold-start problem and gives a clear rationale for the design.

---

## 3. Data Sources

### MovieLens (`ml-latest-small`)
- `ratings.csv` — userId, movieId, rating, timestamp
- `movies.csv` — movieId, title (includes year), genres (pipe-separated)
- `links.csv` — movieId → imdbId, tmdbId (bridge to TMDB)

### TMDB API (deferred — to be added later)
- Will provide: overview text, cast, keywords, poster images, vote average
- Needed for richer content-based features and a nicer UI
- Requires a free API key from themoviedb.org

---

## 4. Data Exploration Findings

| Check | Result |
|---|---|
| Unique users | 610 |
| Unique movies (in movies.csv) | 9,742 |
| Unique movies actually rated | 9,724 |
| Total ratings | ~100,836 |
| Min ratings per user | 20 (guaranteed floor in this dataset) |
| Nulls in ratings.csv | None |
| Nulls in movies.csv | None |
| Rating scale | 0.5–5.0 |
| Movies with only 1 rating | 3,446 |
| Movies with <5 ratings | 6,074 |
| Movies with <10 ratings | 7,455 |
| Movies with "(no genres listed)" | 34 |

**Key takeaway:** rating distribution has a long tail — most movies are rated by very few users. This affects how collaborative filtering is trained and served.

---

## 5. Data Cleaning Decisions

1. **Title parsing**: Extract the year from `"Toy Story (1995)"` into a separate `year` column via regex; keep a `clean_title` without the year.
2. **Genre parsing**: Convert pipe-separated genres (`Action|Adventure`) into space-separated tokens (`"Action Adventure"`) for TF-IDF. The 34 movies with `"(no genres listed)"` are converted to an empty string rather than treated as a literal genre.
3. **No user filtering needed** — every user already has ≥20 ratings.
4. **No null handling needed** — both files are clean.
5. **Movie filtering strategy**: Do **not** drop sparse movies from training data.
   - Content-based model uses **all 9,724 movies** — it doesn't depend on rating count.
   - Collaborative filtering model is **trained on the full ratings set** (SVD handles sparsity natively).
   - Sparse movies (fewer than 5 ratings) are filtered out only **at serving/recommendation time**, so the system doesn't surface statistically meaningless CF predictions.
6. **Train/test split**: Random 80/20 split on ratings (simple, standard approach for a first working model — chosen over time-based split for now).

---

## 6. Planned Architecture

```
User/Movie ID → [Content-Based Score] ─┐
                                          ├─→ Switching logic → Top-N recommendations
User/Movie ID → [Collaborative Score]  ─┘
```

- **Content-based module**: TF-IDF (or sentence-transformers embeddings) on genres + overview + keywords → cosine similarity → `get_similar_movies(movie_id, n)`
- **Collaborative module**: SVD matrix factorization (via `surprise` library) on the user-item rating matrix → `get_cf_recommendations(user_id, n)`
- **Hybrid logic**: checks user's rating count / movie's rating count → routes to content-based or CF accordingly
- **API layer**: FastAPI, endpoints like `/recommend?user_id=X` and `/similar?movie_id=Y`
- **Frontend**: React — search a movie, view recommendations, display posters via TMDB

---

## 7. Tech Stack

| Layer | Tooling |
|---|---|
| Data processing | pandas, scikit-learn |
| Collaborative filtering | `surprise` (SVD) or `implicit` (ALS) |
| Content-based | TF-IDF (scikit-learn) or sentence-transformers |
| Backend/API | FastAPI |
| Frontend | React |
| External data | TMDB API (posters, overview, cast, keywords) |

---

## 8. Progress So Far

- [x] Downloaded MovieLens `ml-latest-small`
- [x] Inspected raw data (genres, year format, ratings scale, nulls, sparsity)
- [x] Decided all data cleaning rules
- [x] Decided train/test split strategy (random 80/20)
- [x] Built `prep_data.py` — loads, cleans, and splits the data into:
  - `movies_clean.csv` (with `year`, `clean_title`, `genres_clean`)
  - `ratings_train.csv`
  - `ratings_test.csv`
- [ ] TMDB API key setup and metadata merge (deferred)
- [ ] Content-based module (TF-IDF + cosine similarity)
- [ ] Collaborative filtering module (SVD)
- [ ] Hybrid switching logic
- [ ] FastAPI backend
- [ ] React frontend

---

## 9. Next Steps
1. Run `prep_data.py` and confirm output (`movies_clean.csv`, `ratings_train.csv`, `ratings_test.csv`)
2. Set up TMDB API key and merge overview/keywords/cast into `movies_clean.csv`
3. Build the content-based model (TF-IDF + cosine similarity)
4. Build the collaborative filtering model (SVD via `surprise`)
5. Implement hybrid switching logic
6. Wrap in FastAPI, then build the React frontend