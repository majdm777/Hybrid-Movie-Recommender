import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";
const TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w92";

const STAR_VALUES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

function RatePage({ userId, ratedCount, onRatingSubmitted }) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    // Tracks which movieIds were just rated in this session, for instant feedback
    const [justRated, setJustRated] = useState({});

    async function handleSearch(e) {
        e.preventDefault();
        if (!query.trim()) return;
        setError("");
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/movies/search?q=${encodeURIComponent(query)}`);
            if (!res.ok) throw new Error("Search failed");
            const data = await res.json();
            setResults(data.results);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }

    async function handleRate(movieId, rating) {
        setError("");
        try {
            const res = await fetch(`${API_BASE}/ratings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_id: Number(userId), movie_id: movieId, rating }),
            });
            if (!res.ok) throw new Error("Could not submit rating");

            setJustRated((prev) => ({ ...prev, [movieId]: rating }));
            onRatingSubmitted();
        } catch (err) {
            setError(err.message);
        }
    }

    return (
        <div className="rate-page">
            <div className="rate-progress">
                You've rated <strong>{ratedCount}</strong> movie{ratedCount === 1 ? "" : "s"}.
                {ratedCount < 5 && (
                    <span className="rate-progress-hint">
                        {" "}
                        Rate at least {5 - ratedCount} more to unlock personalized recommendations.
                    </span>
                )}
            </div>

            <form onSubmit={handleSearch} className="search-form">
                <input
                    type="text"
                    placeholder="Search for a movie you've watched..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button type="submit">Search</button>
            </form>

            {error && <div className="error-banner">{error}</div>}
            {loading && <p>Searching...</p>}

            <div className="rate-results">
                {results.map((movie) => (
                    <div key={movie.movieId} className="rate-row">
                        {movie.poster_path ? (
                            <img
                                src={`${TMDB_IMG_BASE}${movie.poster_path}`}
                                alt={movie.title}
                                className="rate-poster"
                            />
                        ) : (
                            <div className="rate-poster rate-poster--placeholder">No image</div>
                        )}

                        <div className="rate-row-info">
                            <p className="rate-title">
                                {movie.title} {movie.year ? `(${movie.year})` : ""}
                            </p>

                            <div className="star-row">
                                {STAR_VALUES.map((val) => (
                                    <button
                                        key={val}
                                        type="button"
                                        className={`star-button ${justRated[movie.movieId] === val ? "star-button--active" : ""
                                            }`}
                                        onClick={() => handleRate(movie.movieId, val)}
                                        title={`Rate ${val}`}
                                    >
                                        {val}
                                    </button>
                                ))}
                            </div>

                            {justRated[movie.movieId] && (
                                <p className="rate-confirmation">
                                    Rated {justRated[movie.movieId]} ✓
                                </p>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default RatePage;