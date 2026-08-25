import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.models.database.geist_user import WorkspaceModel, get_default_workspace


logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_current_workspace() -> WorkspaceModel:
    """Return the singleton workspace used by local unauthenticated routes."""
    return get_default_workspace()


async def get_authenticated_workspace(
    token: str | None = Depends(oauth2_scheme),
) -> WorkspaceModel:
    """Authorize the legacy test token and return the singleton workspace.

    Args:
        token (str): The OAuth2 token from the request

    Returns:
        The local workspace associated with this Geist process.

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
        # Compatibility-only test token. The operator-principal PR replaces this
        # endpoint-specific mechanism with a global request principal boundary.
        if token.startswith("test_token_"):
            # Extract user_id safely with proper validation
            parts = token.split("_")
            # Ensure exact format: ["test", "token", "{user_id}"]
            if len(parts) != 3 or parts[0] != "test" or parts[1] != "token":
                raise ValueError("Invalid test token format - expected 'test_token_{user_id}'")
            try:
                workspace_id = int(parts[2])
            except ValueError:
                raise ValueError(f"Invalid workspace_id in token: {parts[2]}") from None
            workspace = get_default_workspace()
            if workspace_id != workspace.workspace_id:
                raise ValueError("Token does not identify the local workspace")
            logger.info("Authorized request for workspace %s", workspace_id)
            return workspace
        else:
            # Handle real tokens here in the future
            raise ValueError("Invalid token format")
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
