from bson import ObjectId
from datetime import datetime
from utils.mongo_client import db

class Test:
    """Test model for managing tests"""
    collection = db['tests']

    @staticmethod
    def calculate_score(test_doc, answers):
        total_score = 0
        questions = test_doc.get('questions', [])

        correct_answers = 0
        wrong_answers = 0
        unanswered = 0
        total_marks = 0

        for question in questions:
            question_id = question.get('id')
            student_answer = answers.get(question_id)
            marks = question.get('marks', 0)

            total_marks += marks

            if student_answer is None or str(student_answer).strip() == '':
                unanswered += 1
                continue

            if question.get('type') == 'multiple-choice':
                correct_answer = question.get('correct_answer')

                # ✅ Normalize BOTH values
                try:
                    student_answer = str(student_answer).strip()
                    correct_answer = str(correct_answer).strip()
                except Exception:
                    wrong_answers += 1
                    continue

                if student_answer == correct_answer:
                    total_score += marks
                    correct_answers += 1
                else:
                    wrong_answers += 1

            elif question.get('type') in ['text', 'essay']:
                # Manual grading later
                unanswered += 1

        percentage = (total_score / total_marks * 100) if total_marks > 0 else 0

        return {
            'score': total_score,
            'total_marks': total_marks,
            'percentage': round(percentage, 2),
            'correct_answers': correct_answers,
            'wrong_answers': wrong_answers,
            'unanswered': unanswered,
            'total_questions': len(questions)
        }


    
    # @staticmethod
    # def calculate_score(test_doc, answers):
    #     """
    #     Calculate score based on student answers
        
    #     Args:
    #         test_doc: Test document from MongoDB
    #         answers: Dictionary of {question_id: answer}
        
    #     Returns:
    #         Dictionary with score details
    #     """
    #     total_score = 0
    #     total_marks = test_doc.get('total_marks', 0)
    #     questions = test_doc.get('questions', [])
        
    #     correct_answers = 0
    #     wrong_answers = 0
    #     unanswered = 0
        
    #     for question in questions:
    #         question_id = question.get('id')
    #         student_answer = answers.get(question_id)
            
    #         if student_answer is None or student_answer == '':
    #             unanswered += 1
    #             continue
            
    #         # For multiple choice questions
    #         if question.get('type') == 'multiple-choice':
    #             correct_answer = question.get('correct_answer')
    #             marks = question.get('marks', 0)
                
    #             # Convert student answer to int if it's a string
    #             try:
    #                 student_answer_index = int(student_answer)
    #             except (ValueError, TypeError):
    #                 wrong_answers += 1
    #                 continue
                
    #             if student_answer_index == correct_answer:
    #                 total_score += marks
    #                 correct_answers += 1
    #             else:
    #                 wrong_answers += 1
            
    #         # For text/essay questions - manual grading needed
    #         elif question.get('type') in ['text', 'essay']:
    #             # Store for manual grading later
    #             # For now, mark as unanswered in auto-grading
    #             pass
        
    #     return {
    #         'score': total_score,
    #         'total_marks': total_marks,
    #         'percentage': round((total_score / total_marks * 100), 2) if total_marks > 0 else 0,
    #         'correct_answers': correct_answers,
    #         'wrong_answers': wrong_answers,
    #         'unanswered': unanswered,
    #         'total_questions': len(questions)
    #     }

    

    
    @staticmethod
    def create(title, description, duration, total_marks, questions, created_by, start_date=None, end_date=None):
        """Create a new test"""
        test_doc = {
            'title': title,
            'description': description,
            'duration': duration,  # in minutes
            'total_questions': len(questions),
            'total_marks': total_marks,
            'questions': questions,
            'visibility': 'draft',
            'status': 'available',
            'created_by': created_by,  # admin user_id
            'start_date': start_date,
            'end_date': end_date,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = Test.collection.insert_one(test_doc)
        test_doc['_id'] = result.inserted_id
        return test_doc
    
    @staticmethod
    def find_by_id(test_id):
        """Find test by ID"""
        return Test.collection.find_one({'_id': ObjectId(test_id)})
    
    @staticmethod
    def find_all(filters=None):
        """Find all tests with optional filters"""
        query = filters or {}
        return list(Test.collection.find(query))
    
    @staticmethod
    def update(test_id, update_data):
        """Update test"""
        update_data['updated_at'] = datetime.utcnow()
        result = Test.collection.update_one(
            {'_id': ObjectId(test_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0
    
    @staticmethod
    def delete(test_id):
        """Delete test"""
        result = Test.collection.delete_one({'_id': ObjectId(test_id)})
        return result.deleted_count > 0
    
    @staticmethod
    def publish(test_id):
        """Publish a test"""
        return Test.update(test_id, {'visibility': 'published'})
    
    @staticmethod
    def unpublish(test_id):
        """Unpublish a test"""
        return Test.update(test_id, {'visibility': 'draft'})
    
    @staticmethod
    def get_published_tests():
        """Get all published tests for students"""
        return Test.find_all({'visibility': 'published'})
    
    @staticmethod
    def to_dict(test_doc, include_answers=True):
        """Convert test document to dictionary"""
        if not test_doc:
            return None
        
        # Remove correct answers for students
        questions = test_doc.get('questions', [])
        if not include_answers:
            questions = [
                {
                    'id': q.get('id'),
                    'question': q.get('question'),
                    'type': q.get('type'),
                    'options': q.get('options'),
                    'marks': q.get('marks')
                }
                for q in questions
            ]
        
        return {
            'id': str(test_doc['_id']),
            'title': test_doc.get('title'),
            'description': test_doc.get('description'),
            'duration': test_doc.get('duration'),
            'total_questions': test_doc.get('total_questions'),
            'total_marks': test_doc.get('total_marks'),
            'questions': questions,
            'visibility': test_doc.get('visibility'),
            'status': test_doc.get('status'),
            'start_date': test_doc.get('start_date').isoformat() if test_doc.get('start_date') else None,
            'end_date': test_doc.get('end_date').isoformat() if test_doc.get('end_date') else None,
            'created_at': test_doc.get('created_at').isoformat() if test_doc.get('created_at') else None,
            'updated_at': test_doc.get('updated_at').isoformat() if test_doc.get('updated_at') else None
        }
    

# class TestSession:
#     """TestSession model for tracking student test attempts"""
#     collection = db['test_sessions']  # MongoDB collection name
    
#     @staticmethod
#     def create(test_id, student_id, duration):
#         """
#         Create a new test session when student starts a test
        
#         Args:
#             test_id: ID of the test being taken
#             student_id: ID of the student taking the test
#             duration: Time limit in minutes
        
#         Returns:
#             Created session document
#         """
#         session_doc = {
#             'test_id': ObjectId(test_id),  # Convert string ID to MongoDB ObjectId
#             'student_id': student_id,
#             'start_time': datetime.utcnow(),  # Record when test started
#             'end_time': None,  # Will be set when test ends
#             'duration': duration,  # in minutes (e.g., 60 for 1 hour)
#             'answers': {},  # Will store student's answers as {question_id: answer}
#             'violations': [],  # Will store proctoring violations
#             'risk_score': 0,  # Cumulative score of violations
#             'status': 'in_progress',  # Can be: in_progress, completed, abandoned
#             'submitted_at': None,  # When student submits the test
#             'score': None,  # Final score (calculated after submission)
#             'created_at': datetime.utcnow(),
#             'updated_at': datetime.utcnow()
#         }
        
#         # Insert into MongoDB
#         result = TestSession.collection.insert_one(session_doc)
#         session_doc['_id'] = result.inserted_id  # Add the generated ID
#         return session_doc
    
#     @staticmethod
#     def find_by_id(session_id):
#         """
#         Find a test session by its ID
        
#         Args:
#             session_id: The session ID to search for
        
#         Returns:
#             Session document or None
#         """
#         return TestSession.collection.find_one({'_id': ObjectId(session_id)})
    
#     @staticmethod
#     def find_active_session(test_id, student_id):
#         """
#         Check if student already has an active (in_progress) session for this test
#         This prevents students from starting the same test multiple times
        
#         Args:
#             test_id: ID of the test
#             student_id: ID of the student
        
#         Returns:
#             Active session document or None
#         """
#         return TestSession.collection.find_one({
#             'test_id': ObjectId(test_id),
#             'student_id': student_id,
#             'status': 'in_progress'
#         })
    
#     @staticmethod
#     def update(session_id, update_data):
#         """
#         Update an existing test session
        
#         Args:
#             session_id: ID of session to update
#             update_data: Dictionary of fields to update
        
#         Returns:
#             True if updated, False otherwise
#         """
#         update_data['updated_at'] = datetime.utcnow()
#         result = TestSession.collection.update_one(
#             {'_id': ObjectId(session_id)},
#             {'$set': update_data}
#         )
#         return result.modified_count




class TestSession:
    """TestSession model for tracking student test attempts"""
    collection = db['test_sessions']
    
    @staticmethod
    def create(test_id, student_id, duration):
        """Create a new test session"""
        session_doc = {
            'test_id': ObjectId(test_id),
            'student_id': student_id,
            'start_time': datetime.utcnow(),
            'end_time': None,
            'duration': duration,
            'answers': {},
            'violations': [],
            'risk_score': 0,
            'status': 'in_progress',
            'submitted_at': None,
            'score': None,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = TestSession.collection.insert_one(session_doc)
        session_doc['_id'] = result.inserted_id
        return session_doc
    
    @staticmethod
    def find_by_id(session_id):
        """Find session by ID"""
        return TestSession.collection.find_one({'_id': ObjectId(session_id)})
    
    @staticmethod
    def find_active_session(test_id, student_id):
        """Find active session for student and test"""
        return TestSession.collection.find_one({
            'test_id': ObjectId(test_id),
            'student_id': student_id,
            'status': 'in_progress'
        })
    
    @staticmethod
    def update(session_id, update_data):
        """Update session"""
        update_data['updated_at'] = datetime.utcnow()
        result = TestSession.collection.update_one(
            {'_id': ObjectId(session_id)},
            {'$set': update_data}
        )
        return result.modified_count > 0
    
    @staticmethod
    def add_violation(session_id, violation):
        """Add a proctoring violation"""
        result = TestSession.collection.update_one(
            {'_id': ObjectId(session_id)},
            {
                '$push': {'violations': violation},
                '$inc': {'risk_score': violation.get('severity_score', 10)},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        return result.modified_count > 0
    
    @staticmethod
    def to_dict(session_doc):
        """Convert session document to dictionary"""
        if not session_doc:
            return None
        
        return {
            'id': str(session_doc['_id']),
            'test_id': str(session_doc['test_id']),
            'student_id': session_doc['student_id'],
            'start_time': session_doc['start_time'].isoformat() if session_doc.get('start_time') else None,
            'end_time': session_doc['end_time'].isoformat() if session_doc.get('end_time') else None,
            'duration': session_doc['duration'],
            'answers': session_doc.get('answers', {}),
            'violations': session_doc.get('violations', []),
            'risk_score': session_doc.get('risk_score', 0),
            'status': session_doc['status'],
            'submitted_at': session_doc['submitted_at'].isoformat() if session_doc.get('submitted_at') else None,
            'score': session_doc.get('score'),
        }