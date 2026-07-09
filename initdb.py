import importlib

from app.models.database.database import Base, DATABASE_CONFIG, Engine
from app.models.database.database_config import initialize_database


def main() -> None:
    initialize_database(DATABASE_CONFIG)

    # Import all models to register them with Base.metadata.
    importlib.import_module("app.models.database")

    Base.metadata.create_all(bind=Engine)

    from scripts.insert_presets import main as insert_presets

    insert_presets(to_commit=True, overwrite=False)


if __name__ == "__main__":
    main()
