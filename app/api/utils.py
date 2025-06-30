from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict:
    """Get the current authenticated user from the token.
    
    Args:
        token (str): The OAuth2 token from the request
        
    Returns:
        Dict: User information including user_id and email
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not token:
        logger.error("No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    try:
        # Mock get user implementation for testing
        # Token format: "test_token_{user_id}"
        if token.startswith("test_token_"):
            user_id = int(token.split("_")[-1])  # Extract user_id from token
            logger.info(f"Authenticated test user with ID: {user_id}")
            return {
                "user_id": user_id,
                "email": f"user{user_id}@example.com"
            }
        else:
            # Handle real tokens here in the future
            raise ValueError("Invalid token format")
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) 