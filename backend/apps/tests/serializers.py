from rest_framework import serializers

class QuestionSerializer(serializers.Serializer):
    """Serializer for question"""
    id = serializers.CharField()
    question = serializers.CharField()
    type = serializers.ChoiceField(choices=['multiple-choice', 'text', 'true-false'])
    options = serializers.ListField(child=serializers.CharField(), required=False)
    correct_answer = serializers.CharField()
    marks = serializers.IntegerField()

class TestSerializer(serializers.Serializer):
    """Serializer for creating/updating tests"""
    title = serializers.CharField(max_length=200)
    description = serializers.CharField()
    duration = serializers.IntegerField()
    total_marks = serializers.IntegerField()
    questions = QuestionSerializer(many=True)
    start_date = serializers.DateTimeField(required=False, allow_null=True)
    end_date = serializers.DateTimeField(required=False, allow_null=True)

class TestResponseSerializer(serializers.Serializer):
    """Serializer for test response"""
    id = serializers.CharField(read_only=True)
    title = serializers.CharField()
    description = serializers.CharField()
    duration = serializers.IntegerField()
    total_questions = serializers.IntegerField()
    total_marks = serializers.IntegerField()
    status = serializers.CharField()
    start_date = serializers.DateTimeField(allow_null=True)
    end_date = serializers.DateTimeField(allow_null=True)
    created_at = serializers.DateTimeField()
    updated_at = serializers.DateTimeField()



class SubmitTestSerializer(serializers.Serializer):
    """Serializer for test submission"""
    answers = serializers.DictField(
        child=serializers.CharField(allow_blank=True),
        help_text="Dictionary of question IDs to answers"
    )
    violations = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list,
        help_text="List of proctoring violations"
    )
    risk_score = serializers.IntegerField(
        required=False,
        default=0,
        min_value=0,
        max_value=100,
        help_text="Overall risk score from proctoring"
    )