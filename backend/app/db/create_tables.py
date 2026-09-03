import sys
from pathlib import Path

# Ensure project root is in sys.path so backend module can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import backend.app.db.models  # noqa: F401
from backend.app.db.database import Base, engine


def create_tables():
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")


if __name__ == "__main__":
    create_tables()
