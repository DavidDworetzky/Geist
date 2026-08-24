from app.models.database.geist_user import ensure_default_user


def insert_default_user():
    """Ensure the neutral local workspace identity exists."""
    default_user = ensure_default_user()
    print(f"Default workspace is ready with ID: {default_user.user_id}")

if __name__ == '__main__':
    insert_default_user()
