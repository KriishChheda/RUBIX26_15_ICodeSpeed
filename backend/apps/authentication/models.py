from bson import ObjectId
from datetime import datetime
from django.contrib.auth.hashers import make_password, check_password
from utils.mongo_client import users_collection

class User:
    @staticmethod
    def create(name, email, password, role='student'):
        """Create a new user"""
        if users_collection.find_one({'email': email}):
            raise ValueError('User with this email already exists')
        
        user_data = {
            'name': name,
            'email': email,
            'password': make_password(password),
            'role': role,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = users_collection.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return user_data
    
    @staticmethod
    def find_by_email(email):
        """Find user by email"""
        return users_collection.find_one({'email': email})
    
    @staticmethod
    def find_by_id(user_id):
        """Find user by ID"""
        return users_collection.find_one({'_id': ObjectId(user_id)})
    
    @staticmethod
    def verify_password(stored_password, provided_password):
        """Verify password"""
        return check_password(provided_password, stored_password)
    
    @staticmethod
    def to_dict(user_doc):
        """Convert MongoDB document to dictionary"""
        if not user_doc:
            return None
        
        return {
            'id': str(user_doc['_id']),
            'name': user_doc['name'],
            'email': user_doc['email'],
            'role': user_doc['role'],
            'created_at': user_doc.get('created_at'),
            'updated_at': user_doc.get('updated_at')
        }