from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
)
from pydantic import BaseModel

from app.database import create_user, delete_user, get_user, list_users
from app.security import (
    verify_password,
    create_access_token,
    decode_access_token,
)

router = APIRouter()


# ==========================================================
# MODÈLES
# ==========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "operator"


# ==========================================================
# CONNEXION
# ==========================================================

@router.post("/login")
def login(data: LoginRequest):
    user = get_user(data.username)

    if user is None:
        raise HTTPException(status_code=401, detail="Utilisateur inconnu.")

    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Mot de passe incorrect.")

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"],
    })

    response = Response(
        content='{"message":"Connexion réussie","role":"' + user["role"] + '"}',
        media_type="application/json",
    )

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=1800,
    )

    return response


# ==========================================================
# VÉRIFICATION DU COOKIE JWT
# ==========================================================

def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")

    if token is None:
        raise HTTPException(status_code=401, detail="Utilisateur non authentifié.")

    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(status_code=401, detail="Session expirée ou invalide.")

    return payload


def require_admin(request: Request) -> dict:
    user = get_current_user(request)
    if user.get("role") != "administrator":
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs.")
    return user


# ==========================================================
# PROFIL
# ==========================================================

@router.get("/me")
def me(request: Request):
    return get_current_user(request)


# ==========================================================
# GESTION DES UTILISATEURS (admin uniquement)
# ==========================================================

@router.get("/api/users")
def get_users(request: Request):
    require_admin(request)
    return list_users()


@router.post("/api/users")
def add_user(data: CreateUserRequest, request: Request):
    require_admin(request)
    try:
        user = create_user(data.username, data.password, data.role)
        return {"message": f"Utilisateur '{data.username}' créé.", "user": user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/users/{username}")
def remove_user(username: str, request: Request):
    require_admin(request)
    try:
        deleted = delete_user(username)
        if not deleted:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
        return {"message": f"Utilisateur '{username}' supprimé."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================================
# DÉCONNEXION
# ==========================================================

@router.post("/logout")
def logout():
    response = Response(
        content='{"message":"Déconnexion réussie"}',
        media_type="application/json",
    )
    response.delete_cookie("access_token")
    return response
