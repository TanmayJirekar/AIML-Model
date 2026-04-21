import sqlite3
import pandas as pd
import os

DB_NAME = "career_intelligence.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            experience REAL,
            education_level INTEGER,
            age REAL,
            certifications INTEGER,
            projects INTEGER,
            predicted_salary REAL,
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(experience, education, age, certifications, projects, salary, model_used="Random Forest"):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (experience, education_level, age, certifications, projects, predicted_salary, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (experience, education, age, certifications, projects, salary, model_used))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB Insert Error: {e}")
        return False

def get_all_predictions():
    try:
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"DB Read Error: {e}")
        return pd.DataFrame()

# Initialize upon import
init_db()
