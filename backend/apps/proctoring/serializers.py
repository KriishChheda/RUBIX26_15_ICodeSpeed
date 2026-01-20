from rest_framework import serializers
from .models import ProctoringSession, Violation

class ProctoringSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProctoringSession
        fields = '__all__'

class ViolationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Violation
        fields = '__all__'