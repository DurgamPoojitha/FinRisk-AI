"""
FastAPI Authentication & RBAC Dependencies.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.auth import User
from backend.app.auth.security import decode_access_token


class AuthenticatedUser:
    def __init__(self, user_id: str, email: str, role: str, tenant_id: str):
        self.id = user_id
        self.email = email
        self.role = role
        self.tenant_id = tenant_id


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> AuthenticatedUser:
    """
    Extracts Bearer token from Authorization header and verifies claims.
    If no Authorization header is provided, defaults to a standard demo analyst user
    to maintain seamless local developer and Streamlit prototype compatibility.
    """
    if not authorization or not authorization.startswith("Bearer "):
        # Default local sandbox user
        return AuthenticatedUser(
            user_id="usr_demo_001",
            email="risk_analyst@finrisk.ai",
            role="risk_manager",
            tenant_id="default_tenant",
        )

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or tampered access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(
        user_id=payload.get("sub", "usr_unknown"),
        email=payload.get("email", ""),
        role=payload.get("role", "analyst"),
        tenant_id=payload.get("tenant_id", "default_tenant"),
    )


def require_role(allowed_roles: List[str]):
    """Role-Based Access Control (RBAC) dependency factory."""
    def role_checker(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if current_user.role not in allowed_roles and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Role '{current_user.role}' lacks permission. Required: {allowed_roles}",
            )
        return current_user
    return role_checker
