from app.models.database.geist_user import GeistUser
from app.models.database.database import SessionLocal

def insert_default_user():
    """Insert default user (David Dworetzky) if not already present."""
    
    with SessionLocal() as session:
        # Check if user already exists
        existing_user = session.query(GeistUser).filter_by(email='david@phantasmal.ai').first()
        
        if existing_user is None:
            # Create new user
            default_user = GeistUser(
                name='David Dworetzky',
                email='david@phantasmal.ai'
            )
                    
            # Add and commit to database
            session.add(default_user)
            session.commit()
            print(f"Default user David Dworetzky created successfully with ID: {default_user.user_id}")
        else:
            print(f"Default user already exists with ID: {existing_user.user_id}")

if __name__ == '__main__':
    insert_default_user() 