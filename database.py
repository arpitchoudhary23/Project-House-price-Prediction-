import sqlite3

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        area REAL,
        bedrooms REAL,
        bathrooms REAL,
        floors REAL,
        yearbuilt REAL,
        predicted_price REAL
    )
    """)

    conn.commit()
    conn.close()