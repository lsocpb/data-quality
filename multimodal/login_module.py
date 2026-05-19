def validate_user_identifier(user_id: str) -> str:
    user_id = user_id.strip()

    if not user_id:
        raise ValueError("Identyfikator użytkownika nie może być pusty.")

    return user_id