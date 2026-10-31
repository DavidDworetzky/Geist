from app.models.database.geist_user import ensure_default_workspace


def insert_default_user():
    """Ensure the neutral local workspace identity exists."""
    workspace = ensure_default_workspace()
    print(f"Default workspace is ready with ID: {workspace.workspace_id}")

if __name__ == '__main__':
    insert_default_user()
