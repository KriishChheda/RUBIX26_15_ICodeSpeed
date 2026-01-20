from rest_framework import generics, permissions
from .models import User
from .serializers import UserSerializer, LoginSerializer, SignupSerializer
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken

def get_tokens_for_user(user_doc):
    """Generate JWT tokens for user"""
    refresh = RefreshToken()
    refresh['user_id'] = str(user_doc['_id'])
    refresh['email'] = user_doc['email']
    refresh['role'] = user_doc['role']
    
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """Login endpoint"""
    serializer = LoginSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'error': 'Invalid input', 'details': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    email = serializer.validated_data['email']
    password = serializer.validated_data['password']
    
    # Find user
    user_doc = User.find_by_email(email)
    if not user_doc:
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # Verify password
    if not User.verify_password(user_doc['password'], password):
        return Response(
            {'error': 'Invalid credentials'},
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # Generate tokens
    tokens = get_tokens_for_user(user_doc)
    user_data = User.to_dict(user_doc)
    user_data['token'] = tokens['access']
    
    # return Response(user_data, status=status.HTTP_200_OK)
    return Response({
        'user': user_data,
        'token': tokens['access']
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    """Signup endpoint"""
    serializer = SignupSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {'error': 'Invalid input', 'details': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Create user
        user_doc = User.create(
            name=serializer.validated_data['name'],
            email=serializer.validated_data['email'],
            password=serializer.validated_data['password'],
            role=serializer.validated_data.get('role', 'student')
        )
        
        # Generate tokens
        tokens = get_tokens_for_user(user_doc)
        user_data = User.to_dict(user_doc)
        user_data['token'] = tokens['access']
        
        # return Response(user_data, status=status.HTTP_201_CREATED)
        return Response({
            'user': user_data,
            'token': tokens['access']
        }, status=status.HTTP_201_CREATED)
        
    except ValueError as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['POST'])
def logout(request):
    """Logout endpoint"""
    return Response(
        {'message': 'Logged out successfully'},
        status=status.HTTP_200_OK
    )