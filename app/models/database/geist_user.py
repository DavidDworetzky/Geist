import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary, DateTime, Boolean, ARRAY, DateTime
from sqlalchemy.orm import relationship, Session
from app.models.database.database import Base, Session
from sqlalchemy.dialects.postgresql import insert
from app.models.database.database import SessionLocal
from dataclasses import dataclass

import uuid

class GeistUser(Base):
    __tablename__ = "geist_user"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    #hashed password
    password = Column(String)

@dataclass
class UserModel:
    user_id: int
    username: str
    name: str
    email: str
    password: str


def get_user_by_id(user_id: int) -> UserModel:
    with SessionLocal() as session:
        user = session.query(GeistUser).filter_by(user_id=user_id).first()
        return UserModel(user_id=user.user_id, username=user.username, name=user.name, email=user.email, password=user.password)
    
def create_user(user: UserModel) -> UserModel:
    with SessionLocal() as session:
        session.add(user)
        session.commit()
        return UserModel(user_id=user.user_id, username=user.username, name=user.name, email=user.email, password=user.password)

def get_default_user() -> UserModel:
    '''
    Returns the default user context if auth isn't created yet. 
    '''
    with SessionLocal() as session:
        user = session.query(GeistUser).filter_by(email='david@phantasmal.ai').first()
        
        if user is None:
            # Create the default user if it doesn't exist
            default_user = GeistUser(
                username='ddworetzky',
                name='David Dworetzky',
                email='david@phantasmal.ai',
                password=''  # Empty password for now
            )
            session.add(default_user)
            session.commit()
            session.refresh(default_user)
            user = default_user
            
        return UserModel(user_id=user.user_id, username=user.username, name=user.name, email=user.email, password=user.password)