from django.urls import path
from . import views

urlpatterns = [
    # path('tests', views.student_tests, name='student-tests'),
    path('tests/', views.get_available_tests, name='available-tests'),  # ✅ Add this
     path('tests/<str:test_id>/start', views.start_test, name='start-test'),  # ← ADD THIS LINE
         path('tests/<str:test_id>/submit', views.submit_test, name='submit-test'),

]