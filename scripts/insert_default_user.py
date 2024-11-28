from app.models.database.user import User
from app.models.database.database import SessionLocal

def insert_default_user():
    """Insert default user (David Dworetzky) if not already present."""
    
    with SessionLocal() as session:
        # Check if user already exists
        existing_user = session.query(User).filter_by(email='david@phantasmal.ai').first()
        
        if existing_user is None:
            # Create new user
            default_user = User(
                name='David Dworetzky',
                email='david@phantasmal.ai'
            )
                    
            # Add and commit to database
            session.add(default_user)
            session.commit()
            print("Default user David Dworetzky created successfully")
        else:
            print("Default user already exists")

if __name__ == '__main__':
    insert_default_user() 