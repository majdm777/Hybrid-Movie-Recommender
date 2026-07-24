import { useState, useEffect } from "react";
import "./App.css";
import SignupPage from "./SignupPage";
import "./SignupPage.css";
import RatePage from "./RatePage";
import "./RatePage.css";

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
  const [myUserId, setMyUserId] = useState(null);
  const [checkedStorage, setCheckedStorage] = useState(false);

  useEffect(() => {
    const savedId = localStorage.getItem("userId");
    if (savedId) {
      setMyUserId(savedId);
    }
    setCheckedStorage(true);
  }, []);

  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [similarMovies, setSimilarMovies] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);

  const [activeTab, setActiveTab] = useState("discover");
  const [ratedCount, setRatedCount] = useState(0);

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
      const res = await fetch(`${API_BASE}/similar/${movie.movieId}?n=0.2`);
      if (!res.ok) throw new Error("Could not fetch similar movies");
      const data = await res.json();
      setSimilarMovies(data.recommendations);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleGetUserRecs() {
    setLoading(true);
    setError("");
    setUserRecs(null);
    try {
      const res = await fetch(`${API_BASE}/recommend/${myUserId}?n=10`);
      if (!res.ok) throw new Error("Could not fetch recommendations");
      const data = await res.json();
      setUserRecs(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!checkedStorage) {
    return null; // avoid a flash of the signup page while checking localStorage
  }

  if (!myUserId) {
    return <SignupPage onSignupComplete={(id) => setMyUserId(id)} />;
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎬 Hybrid Movie Recommender</h1>
        <p className="subtitle">Content-based + Collaborative Filtering</p>
        <p className="signed-in-as">
          Signed in as user #{myUserId}{" "}
          <button
            className="link-button"
            onClick={() => {
              localStorage.removeItem("userId");
              localStorage.removeItem("displayName");
              setMyUserId(null);
            }}
          >
            (switch user)
          </button>
        </p>
      </header>

      <nav className="tab-nav">
        <button
          className={`tab-button ${activeTab === "discover" ? "tab-button--active" : ""}`}
          onClick={() => setActiveTab("discover")}
        >
          Discover
        </button>
        <button
          className={`tab-button ${activeTab === "rate" ? "tab-button--active" : ""}`}
          onClick={() => setActiveTab("rate")}
        >
          Rate Movies
        </button>
      </nav>

      {error && <div className="error-banner">{error}</div>}

      {activeTab === "rate" && (
        <RatePage
          userId={myUserId}
          ratedCount={ratedCount}
          onRatingSubmitted={() => setRatedCount((c) => c + 1)}
        />
      )}

      {activeTab === "discover" && (
        <>
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
            <h2>Your personalized recommendations</h2>
            <button onClick={handleGetUserRecs} className="get-recs-button">
              {loading ? "Loading..." : "Get My Recommendations"}
            </button>

            {userRecs && (
              <div>
                <p className="strategy-badge">
                  You have {userRecs.user_rating_count} ratings → using{" "}
                  <strong>{userRecs.strategy}</strong> strategy
                </p>
                {userRecs.note && <p className="rec-note">{userRecs.note}</p>}
                <div className="movie-grid">
                  {userRecs.recommendations.map((m) => (
                    <MovieCard key={m.movieId} movie={m} />
                  ))}
                </div>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

export default App;