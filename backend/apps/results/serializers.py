from rest_framework import serializers
from apps.results.models import TestResult

class TestResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestResult
        fields = '__all__'  # or specify the fields you want to include, e.g., ['id', 'student', 'test', 'score', 'timestamp']