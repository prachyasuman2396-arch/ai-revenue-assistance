import sys
from pathlib import Path

from sqlalchemy import text

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.db.database import engine


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print(result.fetchone()[0])


if __name__ == "__main__":
    test_connection()
