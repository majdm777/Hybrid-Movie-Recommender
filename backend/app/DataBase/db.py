"""
Movie Recommender - Database Layer (SQLite)
-----------------------------------------------
Stores NEW users (people using the app, not the original MovieLens users)
and the ratings they submit. Original MovieLens users/ratings stay in the
CSV files and are never touched - new users get IDs starting above the
highest existing MovieLens userId, so there's no collision.

This file just sets up the DB and gives helper functions. It's imported
by api.py.
"""

import sqlite3
import pandas as pd

DB_PATH = "data/processed/app.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(starting_user_id: int):
    """
    Creates tables if they don't exist. starting_user_id should be
    (max existing MovieLens userId) + 1, so new users never collide
    with the original dataset's userIds.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(display_name)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_id)
        )
    """)

    # Keep track of the next userId to hand out, seeded above the
    # original MovieLens userId range so there's never a collision.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_id_counter (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            next_id INTEGER NOT NULL
        )
    """)
    cur.execute("SELECT next_id FROM user_id_counter WHERE id = 1")
    row = cur.fetchone()
    if row is None:
        cur.execute(
            "INSERT INTO user_id_counter (id, next_id) VALUES (1, ?)",
            (starting_user_id,),
        )

    conn.commit()
    conn.close()


def create_new_user(display_name: str ) -> int:
    """Assigns the next available userId and creates the user row."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT next_id FROM user_id_counter WHERE id = 1")
    new_id = cur.fetchone()["next_id"]

    cur.execute("SELECT display_name FROM users where display_name = ?",(display_name,))
    name = cur.fetchone()

    if name is not None :
        cur.execute("SELECT user_id FROM users WHERE display_name=?",(name["display_name"],))
        id = cur.fetchone()["user_id"]

        return id

    cur.execute(
        "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
        (new_id, display_name),
    )
    cur.execute(
        "UPDATE user_id_counter SET next_id = ? WHERE id = 1", (new_id + 1,)
    )

    conn.commit()
    conn.close()
    return new_id


def add_rating(user_id: int, movie_id: int, rating: float):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ratings (user_id, movie_id, rating)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, movie_id)
        DO UPDATE SET rating = excluded.rating
        """,
        (user_id, movie_id, rating),
    )
    conn.commit()
    conn.close()


def get_ratings_for_user(user_id: int) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT user_id, movie_id, rating FROM ratings WHERE user_id = ?",
        conn,
        params=(user_id,),
    )
    conn.close()
    return df


def user_exists(user_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def get_all_new_ratings() -> pd.DataFrame:
    """All ratings submitted by new (app) users - used when retraining SVD."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    return df