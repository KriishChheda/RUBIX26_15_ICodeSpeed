from pymongo import MongoClient
from django.conf import settings
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'ai_proctor_db')

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
users_collection = db['users']
tests_collection = db['tests']
test_sessions_collection = db['test_sessions']
proctoring_logs_collection = db['proctoring_logs']
results_collection = db['results']

def get_collection(name):
    return db[name]