from django.db import models

class TestResult(models.Model):
    student_id = models.CharField(max_length=255)
    test_id = models.CharField(max_length=255)
    score = models.FloatField()
    submitted_at = models.DateTimeField(auto_now_add=True)
    violations = models.JSONField(default=dict)

    class Meta:
        indexes = [
            models.Index(fields=['student_id']),
            models.Index(fields=['test_id']),
        ]

    def __str__(self):
        return f"Result for Student {self.student_id} in Test {self.test_id}"