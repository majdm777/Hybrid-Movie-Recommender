import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function SignupPage({ onSignupComplete }) {
    const [displayName, setDisplayName] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    async function handleSignup(e) {
        e.preventDefault();
        setLoading(true);
        setError("");

        try {
            const res = await fetch(`${API_BASE}/users/signup`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ display_name: displayName || null }),
            });

            if (!res.ok) throw new Error("Signup failed. Please try again.");

            const data = await res.json();
            const newUserId = data.user_id;
            // if (newUserId == -1) throw new Error("UserName Already Exist. try another one ")

            // Persist so the user stays "logged in" across refreshes
            localStorage.setItem("userId", newUserId);
            if (displayName) {
                localStorage.setItem("displayName", displayName);
            }

            onSignupComplete(newUserId);
        } catch (err) {
            setError(err.message);
        }
        finally {
            setLoading(false);
        }
    }

    return (
        <div className="signup-page">
            <div className="signup-card">
                <h1>🎬 Welcome</h1>
                <p className="signup-subtitle">
                    Create your profile to get movie recommendations tailored just for you.
                </p>

                <form onSubmit={handleSignup} className="signup-form">
                    <label htmlFor="displayName">Your UserName</label>
                    <input
                        id="displayName"
                        type="text"
                        placeholder="e.g. name"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                    />

                    {error && <p className="signup-error">{error}</p>}

                    <button type="submit" disabled={loading}>
                        {loading ? "Creating your profile..." : "Get Started"}
                    </button>
                </form>
            </div>
        </div>
    );
}

export default SignupPage;