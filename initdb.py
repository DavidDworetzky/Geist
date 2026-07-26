from app.database_upgrade import upgrade_database


def main() -> None:
    upgrade_database()


if __name__ == "__main__":
    main()
