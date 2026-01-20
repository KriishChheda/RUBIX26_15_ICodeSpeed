from django.utils.crypto import get_random_string
import jwt
from datetime import datetime, timedelta
from django.conf import settings

def generate_token(user):
    payload = {
        'user_id': str(user.id),
        'exp': datetime.utcnow() + timedelta(days=1),  # Token expires in 1 day
        'iat': datetime.utcnow()
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')
    return token

def decode_token(token):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def generate_random_password(length=8):
    return get_random_string(length)