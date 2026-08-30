from fastapi import Depends, HTTPException, status

from app.models.database.geist_user import WorkspaceModel, get_default_workspace
from app.security.operator import OperatorPrincipal, get_operator_principal


def get_current_workspace(
    principal: OperatorPrincipal = Depends(get_operator_principal),
) -> WorkspaceModel:
    """Return the workspace authorized by the request principal."""
    workspace = get_default_workspace()
    if principal.workspace_id != workspace.workspace_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied")
    return workspace
