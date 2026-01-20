from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_all_results, name='all-results'),
    path('student/<str:student_id>/', views.get_student_results, name='student-results'),
    path('test/<str:test_id>/', views.get_test_results, name='test-results'),

]