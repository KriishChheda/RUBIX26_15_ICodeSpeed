from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import Test, TestSession
from .serializers import TestSerializer, TestResponseSerializer, SubmitTestSerializer
from bson import ObjectId
from datetime import datetime

def is_admin(request):
    """Check if user is admin"""
    return hasattr(request.user, 'role') and request.user.role == 'admin'

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def tests_list(request):
    """
    GET: Get all tests (admin only)
    POST: Create new test (admin only)
    """
    if not is_admin(request):
        return Response(
            {'error': 'Admin access required'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    if request.method == 'GET':
        tests = Test.find_all()
        tests_data = [Test.to_dict(test) for test in tests]
        return Response(tests_data, status=status.HTTP_200_OK)
    
    elif request.method == 'POST':
        serializer = TestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid input', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user_id = request.user.id  # Now this is the MongoDB ObjectId string
            
            test_doc = Test.create(
                title=serializer.validated_data['title'],
                description=serializer.validated_data['description'],
                duration=serializer.validated_data['duration'],
                total_marks=serializer.validated_data['total_marks'],
                questions=serializer.validated_data['questions'],
                created_by=user_id,
                start_date=serializer.validated_data.get('start_date'),
                end_date=serializer.validated_data.get('end_date')
            )
            
            return Response(
                Test.to_dict(test_doc),
                status=status.HTTP_201_CREATED
            )
        
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

# @api_view(['GET', 'PUT', 'DELETE'])
# @permission_classes([IsAuthenticated])
# def test_detail(request, test_id):
#     """
#     GET: Get test by ID
#     PUT: Update test (admin only)
#     DELETE: Delete test (admin only)
#     """
#     try:
#         test_doc = Test.find_by_id(test_id)
        
#         if not test_doc:
#             return Response(
#                 {'error': 'Test not found'},
#                 status=status.HTTP_404_NOT_FOUND
#             )
        
#         if request.method == 'GET':
#             # Students can view published tests without answers
#             include_answers = is_admin(request)
#             return Response(
#                 Test.to_dict(test_doc, include_answers=include_answers),
#                 status=status.HTTP_200_OK
#             )
        
#         # Admin-only actions
#         if not is_admin(request):
#             return Response(
#                 {'error': 'Admin access required'},
#                 status=status.HTTP_403_FORBIDDEN
#             )
        
#         if request.method == 'PUT':
#             serializer = TestSerializer(data=request.data)
            
#             if not serializer.is_valid():
#                 return Response(
#                     {'error': 'Invalid input', 'details': serializer.errors},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
            
#             update_data = {
#                 'title': serializer.validated_data['title'],
#                 'description': serializer.validated_data['description'],
#                 'duration': serializer.validated_data['duration'],
#                 'total_marks': serializer.validated_data['total_marks'],
#                 'questions': serializer.validated_data['questions'],
#                 'total_questions': len(serializer.validated_data['questions']),
#                 'start_date': serializer.validated_data.get('start_date'),
#                 'end_date': serializer.validated_data.get('end_date')
#             }
            
#             Test.update(test_id, update_data)
#             updated_test = Test.find_by_id(test_id)
            
#             return Response(
#                 Test.to_dict(updated_test),
#                 status=status.HTTP_200_OK
#             )
        
#         elif request.method == 'DELETE':
#             Test.delete(test_id)
#             return Response(
#                 {'message': 'Test deleted successfully'},
#                 status=status.HTTP_200_OK
#             )
    
#     except Exception as e:
#         return Response(
#             {'error': str(e)},
#             status=status.HTTP_400_BAD_REQUEST
#         )


@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def test_detail(request, test_id):
    """
    GET: Get test by ID
    PUT: Update test (admin only)
    DELETE: Delete test (admin only)
    """
    try:
        test_doc = Test.find_by_id(test_id)
        
        if not test_doc:
            return Response(
                {'error': 'Test not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if request.method == 'GET':
            # Students can view published tests without answers
            include_answers = is_admin(request)
            return Response(
                Test.to_dict(test_doc, include_answers=include_answers),
                status=status.HTTP_200_OK
            )
        
        # Admin-only actions
        if not is_admin(request):
            return Response(
                {'error': 'Admin access required'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if request.method == 'PUT':
            # Partial update - only update provided fields
            update_data = {}
            
            if 'title' in request.data:
                update_data['title'] = request.data['title']
            if 'description' in request.data:
                update_data['description'] = request.data['description']
            if 'duration' in request.data:
                update_data['duration'] = request.data['duration']
            if 'total_marks' in request.data:
                update_data['total_marks'] = request.data['total_marks']
            if 'questions' in request.data:
                update_data['questions'] = request.data['questions']
                update_data['total_questions'] = len(request.data['questions'])
            if 'start_date' in request.data:
                update_data['start_date'] = request.data['start_date']
            if 'end_date' in request.data:
                update_data['end_date'] = request.data['end_date']
            
            if not update_data:
                return Response(
                    {'error': 'No fields to update'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            Test.update(test_id, update_data)
            updated_test = Test.find_by_id(test_id)
            
            return Response(
                Test.to_dict(updated_test),
                status=status.HTTP_200_OK
            )
        
        elif request.method == 'DELETE':
            Test.delete(test_id)
            return Response(
                {'message': 'Test deleted successfully'},
                status=status.HTTP_200_OK
            )
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )



@api_view(['POST'])
@permission_classes([IsAuthenticated])
def publish_test(request, test_id):
    """Publish or unpublish a test (admin only)"""
    if not is_admin(request):
        return Response(
            {'error': 'Admin access required'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    try:
        test_doc = Test.find_by_id(test_id)
        
        if not test_doc:
            return Response(
                {'error': 'Test not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Toggle publish status
        if test_doc.get('visibility') == 'published':
            Test.unpublish(test_id)
            message = 'Test unpublished successfully'
        else:
            Test.publish(test_id)
            message = 'Test published successfully'
        
        updated_test = Test.find_by_id(test_id)
        
        return Response(
            {
                'message': message,
                'test': Test.to_dict(updated_test)
            },
            status=status.HTTP_200_OK
        )
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def student_tests(request):
    """Get all published tests for students"""
    try:
        tests = Test.get_published_tests()
        # Don't include correct answers for students
        tests_data = [Test.to_dict(test, include_answers=False) for test in tests]
        return Response(tests_data, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_test(request, test_id):
    """
    Start a test - creates a test session for the student
    
    Flow:
    1. Verify test exists and is published
    2. Check if student already has an active session
    3. Create new session
    4. Return test questions (without correct answers) and session info
    
    Args:
        request: HTTP request object (contains authenticated user)
        test_id: ID of the test to start (from URL parameter)
    
    Returns:
        201: Test session created successfully
        400: Validation error (test not available, already started, etc.)
        404: Test not found
    """
    try:
        # STEP 1: Verify test exists
        test_doc = Test.find_by_id(test_id)
        
        if not test_doc:
            return Response(
                {'error': 'Test not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # STEP 2: Check if test is published
        # Only published tests can be started by students
        if test_doc.get('visibility') != 'published':
            return Response(
                {'error': 'Test is not available. It may not be published yet.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # STEP 3: Check if student already has an active session
        # This prevents students from starting the same test multiple times
        student_id = request.user.id  # Get student ID from JWT token
        existing_session = TestSession.find_active_session(test_id, student_id)
        
        if existing_session:
            return Response(
                {
                    'error': 'You already have an active session for this test',
                    'session': TestSession.to_dict(existing_session)
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # STEP 4: Create new test session
        # This marks the official start time
        session_doc = TestSession.create(
            test_id=test_id,
            student_id=student_id,
            duration=test_doc['duration']
        )
        
        # STEP 5: Return response
        # - Session info (so frontend knows the session ID for submitting answers)
        # - Test questions (WITHOUT correct answers for security)
        return Response(
            {
                'message': 'Test session started successfully',
                'session': TestSession.to_dict(session_doc),
                'test': Test.to_dict(test_doc, include_answers=False)  # Don't send correct answers!
            },
            status=status.HTTP_201_CREATED
        )
    
    except Exception as e:
        # Catch any unexpected errors
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
    

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_test(request, test_id):
    """
    Submit test answers and calculate score
    
    Flow:
    1. Validate test session exists and is active
    2. Validate submitted answers
    3. Calculate score for auto-gradable questions
    4. Update session with answers, score, and violations
    5. Mark session as completed
    6. Return results
    
    Args:
        request: HTTP request with answers, violations, risk_score
        test_id: ID of the test being submitted
    
    Returns:
        200: Test submitted successfully with score
        400: Validation error
        404: Test or session not found
    """
    try:
        # Validate request data
        serializer = SubmitTestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid submission data', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        validated_data = serializer.validated_data
        answers = validated_data.get('answers', {})
        violations = validated_data.get('violations', [])
        risk_score = validated_data.get('risk_score', 0)
        
        # Get student ID from JWT
        student_id = request.user.id
        
        # Find active test session
        session_doc = TestSession.find_active_session(test_id, student_id)
        
        if not session_doc:
            return Response(
                {'error': 'No active test session found. Please start the test first.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get test document
        test_doc = Test.find_by_id(test_id)
        
        if not test_doc:
            return Response(
                {'error': 'Test not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Calculate score
        score_data = Test.calculate_score(test_doc, answers)
        
        # Update session with submission data
        submission_time = datetime.utcnow()
        update_data = {
            'answers': answers,
            'violations': violations,
            'risk_score': risk_score,
            'score': score_data['score'],
            'status': 'completed',
            'submitted_at': submission_time,
            'end_time': submission_time,
            'score_details': score_data
        }
        
        TestSession.update(str(session_doc['_id']), update_data)
        
        # Return results
        return Response(
            {
                'message': 'Test submitted successfully',
                'results': {
                    'score': score_data['score'],
                    'total_marks': score_data['total_marks'],
                    'percentage': score_data['percentage'],
                    'correct_answers': score_data['correct_answers'],
                    'wrong_answers': score_data['wrong_answers'],
                    'unanswered': score_data['unanswered'],
                    'total_questions': score_data['total_questions'],
                    'submitted_at': submission_time.isoformat(),
                    'risk_score': risk_score,
                    'violations_count': len(violations)
                }
            },
            status=status.HTTP_200_OK
        )
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

# Add these new views to your existing views.py

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_all_results(request):
    """Get all test results (admin only)"""
    if not is_admin(request):
        return Response(
            {'error': 'Admin access required'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    try:
        # Get all completed sessions
        sessions = list(TestSession.collection.find({'status': 'completed'}))
        
        results = []
        for session in sessions:
            # Get test details
            test = Test.find_by_id(str(session['test_id']))
            
            result = {
                'id': str(session['_id']),
                'student_id': session['student_id'],
                'test_id': str(session['test_id']),
                'test_title': test['title'] if test else 'Unknown Test',
                'score': session.get('score', 0),
                'total_marks': test['total_marks'] if test else 0,
                'violations': session.get('violations', []),
                'risk_score': session.get('risk_score', 0),
                'submitted_at': session.get('submitted_at').isoformat() if session.get('submitted_at') else None,
                'duration': session.get('duration', 0)
            }
            results.append(result)
        
        return Response(results, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_student_results(request, student_id):
    """Get all results for a specific student"""
    # Students can only view their own results, admins can view anyone's
    if not is_admin(request) and str(request.user.id) != student_id:
        return Response(
            {'error': 'You can only view your own results'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    try:
        # Get all completed sessions for this student
        sessions = list(TestSession.collection.find({
            'student_id': student_id,
            'status': 'completed'
        }))
        
        results = []
        for session in sessions:
            # Get test details
            test = Test.find_by_id(str(session['test_id']))
            
            result = {
                'id': str(session['_id']),
                'test_id': str(session['test_id']),
                'test_title': test['title'] if test else 'Unknown Test',
                'score': session.get('score', 0),
                'total_marks': test['total_marks'] if test else 0,
                'percentage': (session.get('score', 0) / test['total_marks'] * 100) if test else 0,
                'violations': session.get('violations', []),
                'risk_score': session.get('risk_score', 0),
                'submitted_at': session.get('submitted_at').isoformat() if session.get('submitted_at') else None,
                'duration': session.get('duration', 0)
            }
            results.append(result)
        
        return Response(results, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_test_results(request, test_id):
    """Get all results for a specific test (admin only)"""
    if not is_admin(request):
        return Response(
            {'error': 'Admin access required'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    try:
        # Get all completed sessions for this test
        sessions = list(TestSession.collection.find({
            'test_id': ObjectId(test_id),
            'status': 'completed'
        }))
        
        # Get test details
        test = Test.find_by_id(test_id)
        
        if not test:
            return Response(
                {'error': 'Test not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        results = []
        for session in sessions:
            result = {
                'id': str(session['_id']),
                'student_id': session['student_id'],
                'score': session.get('score', 0),
                'total_marks': test['total_marks'],
                'percentage': (session.get('score', 0) / test['total_marks'] * 100),
                'violations': session.get('violations', []),
                'risk_score': session.get('risk_score', 0),
                'submitted_at': session.get('submitted_at').isoformat() if session.get('submitted_at') else None,
                'duration': session.get('duration', 0)
            }
            results.append(result)
        
        return Response({
            'test': Test.to_dict(test),
            'results': results
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_available_tests(request):
    """Get all available tests for student with their completion status"""
    try:
        student_id = str(request.user.id)
        
        # Get all published tests
        tests = list(Test.collection.find({'visibility': 'published'}))
        
        result = []
        for test in tests:
            test_id = str(test['_id'])
            
            # Check if student has a session for this test
            session = TestSession.collection.find_one({
                'test_id': test['_id'],
                'student_id': student_id
            })
            
            # Determine status and score
            if session:
                if session.get('status') == 'completed':
                    status_value = 'completed'
                    score = session.get('score', 0)
                elif session.get('status') == 'in_progress':
                    status_value = 'in_progress'
                    score = None
                else:
                    status_value = 'available'
                    score = None
            else:
                status_value = 'available'
                score = None
            
            test_data = {
                'id': test_id,
                'title': test['title'],
                'description': test.get('description', ''),
                'duration': test['duration'],
                'total_marks': test['total_marks'],
                'total_questions': test.get('total_questions', len(test.get('questions', []))),
                'start_date': test.get('start_date'),
                'end_date': test.get('end_date'),
                'status': status_value,  # ✅ Includes completion status
                'score': score,          # ✅ Includes score if completed
                'session_id': str(session['_id']) if session else None
            }
            
            result.append(test_data)
        
        return Response(result, status=status.HTTP_200_OK)
    
    except Exception as e:
        print(f"Error in get_available_tests: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )