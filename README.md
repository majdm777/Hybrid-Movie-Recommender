# Hybrid Movie Recommender

A movie recommendation application that combines content-based filtering and collaborative filtering.

## Project layout

- `backend/app/` contains the FastAPI service and recommendation scripts.
- `backend/data/` contains local MovieLens data, generated datasets, and the TMDB cache. It is intentionally not committed.
- `frontend/` contains the React and Vite user interface.

## Run locally

1. Create and activate a Python virtual environment in `backend/.venv`, then install the backend dependencies:

   ```powershell
   cd backend
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add a TMDB API key if you need to refresh TMDB metadata.

3. Start the API:

   ```powershell
   cd backend
   .\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
   ```

4. In another terminal, start the frontend:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

The API is available at `http://127.0.0.1:8000`, including interactive documentation at `/docs`.
