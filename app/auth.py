from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
)
from pydantic import BaseModel

from database import get_user
from security import (
    verify_password,
    create_access_token,
    decode_access_token,
)

router = APIRouter()


# ==========================================================
# MODELES
# ==========================================================

class LoginRequest(BaseModel):
    username: str
    password: str


# ==========================================================
# CONNEXION
# ==========================================================

@router.post("/login")
def login(data: LoginRequest):

    user = get_user(data.username)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Utilisateur inconnu."
        )

    if not verify_password(
        data.password,
        user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Mot de passe incorrect."
        )

    token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })

    response = Response(
        content="""
{
    "message":"Connexion réussie"
}
""",
        media_type="application/json"
    )

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,      # Passer à True en HTTPS
        max_age=1800
    )

    return response


# ==========================================================
# VERIFICATION DU COOKIE JWT
# ==========================================================

def get_current_user(request: Request):

    token = request.cookies.get("access_token")

    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Utilisateur non authentifié."
        )

    payload = decode_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Session expirée ou invalide."
        )

    return payload


# ==========================================================
# TEST
# ==========================================================

@router.get("/me")
def me(request: Request):

    return get_current_user(request)


# ==========================================================
# DECONNEXION
# ==========================================================

@router.post("/logout")
def logout():

    response = Response(
        content="""
{
    "message":"Déconnexion réussie"
}
""",
        media_type="application/json"
    )

    response.delete_cookie("access_token")

    return response