from django.urls import path
from . import views

urlpatterns = [
    # Admin routes
    path('', views.tests_list, name='tests-list'),
    path('<str:test_id>', views.test_detail, name='test-detail'),
    path('<str:test_id>/publish', views.publish_test, name='publish-test'),
    
]