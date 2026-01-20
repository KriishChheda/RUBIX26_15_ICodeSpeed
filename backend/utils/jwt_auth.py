from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken

class MongoUser:
    """
    Custom user class for MongoDB authentication
    """
    def __init__(self, user_id, email, role):
        self.id = user_id
        self.email = email
        self.role = role
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
        
    def __str__(self):
        return f"{self.email} ({self.role})"

class MongoJWTAuthentication(JWTAuthentication):
    """
    Custom JWT authentication that works with MongoDB
    Instead of querying Django's User model, we trust the JWT payload
    """
    
    def get_user(self, validated_token):
        """
        Returns a user object from the validated token.
        Since we're using MongoDB, we create a simple user object from token data.
        """
        try:
            user_id = validated_token.get('user_id')
            email = validated_token.get('email')
            role = validated_token.get('role')
            
            if not user_id:
                raise InvalidToken('Token contained no recognizable user identification')
            
            # Create custom user object
            return MongoUser(user_id=user_id, email=email, role=role)
            
        except KeyError:
            raise InvalidToken('Token contained no recognizable user identification')