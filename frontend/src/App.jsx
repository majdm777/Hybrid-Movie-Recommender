import { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200";

function MovieCard({ movie }) {
  return (
    <div className="movie-card">
      {movie.poster_path ? (
        <img
          src={`${TMDB_IMG_BASE}${movie.poster_path}`}
          alt={movie.title}
          className="movie-poster"
        />
      ) : (
        <div className="movie-poster movie-poster--placeholder">No image</div>
      )}
      <div className="movie-info">
        <p className="movie-title">{movie.title}</p>
        <p className="movie-meta">
          {movie.year ? movie.year : ""} {movie.score !== undefined ? `· score ${movie.score}` : ""}
        </p>
      </div>
    </div>
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);

  const [userId, setUserId] = useState("");
  const [userRecs, setUserRecs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setError("");
    try {
      const res = await fetch(`${API_BASE}/movies/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error("Search failed");
      const data = await res.json();
      setSearchResults(data.results);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSelectMovie(movie) {
    setSelectedMovie(movie);
    setSimilarMovies([]);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/similar/${movie.movieId}?n=0.20`);
      if (!res.ok) throw new Error("Could not fetch similar movies");
      const data = await res.json();
      setSimilarMovies(data.recommendations);
    } catch (err) {
      setError("error in handleSelectiveError", movie.movieId);
    }
  }

  async function handleGetUserRecs(e) {
    e.preventDefault();
    if (!userId.trim()) return;
    setLoading(true);
    setError("");
    setUserRecs(null);
    try {
      const res = await fetch(`${API_BASE}/recommend/${userId}?n=10`);
      if (!res.ok) throw new Error("Could not fetch recommendations");
      const data = await res.json();
      setUserRecs(data);
    } catch (err) {
      setError("error in HandleGetUserRecs");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎬 Hybrid Movie Recommender</h1>
        <p className="subtitle">Content-based + Collaborative Filtering</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* --- Search & similar movies --- */}
      <section className="section">
        <h2>Find similar movies</h2>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            placeholder="Search a movie title..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit">Search</button>
        </form>

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map((m) => (
              <button
                key={m.movieId}
                className="search-result-item"
                onClick={() => handleSelectMovie(m)}
              >
                {m.title} {m.year ? `(${m.year})` : ""}
              </button>
            ))}
          </div>
        )}

        {selectedMovie && (
          <div className="similar-section">
            <h3>Movies similar to "{selectedMovie.title}"</h3>
            <div className="movie-grid">
              {similarMovies.map((m) => (
                <MovieCard key={m.movieId} movie={m} />
              ))}
            </div>
          </div>
        )}
      </section>

      {/* --- Personalized recommendations --- */}
      <section className="section">
        <h2>Personalized recommendations</h2>
        <form onSubmit={handleGetUserRecs} className="search-form">
          <input
            type="number"
            placeholder="Enter a userId (e.g. 1)"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
          <button type="submit">Get Recommendations</button>
        </form>

        {loading && <p>Loading...</p>}

        {userRecs && (
          <div>
            <p className="strategy-badge">
              User has {userRecs.user_rating_count} ratings → using{" "}
              <strong>{userRecs.strategy}</strong> strategy
            </p>
            <div className="movie-grid">
              {userRecs.recommendations.map((m) => (
                <MovieCard key={m.movieId} movie={m} />
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;