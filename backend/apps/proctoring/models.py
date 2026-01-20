from django.db import models

class ProctoringSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    student_id = models.CharField(max_length=255)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    risk_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"Session {self.session_id} for Student {self.student_id}"

class Violation(models.Model):
    session = models.ForeignKey(ProctoringSession, related_name='violations', on_delete=models.CASCADE)
    violation_type = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    description = models.TextField()

    def __str__(self):
        return f"Violation {self.violation_type} at {self.timestamp}"