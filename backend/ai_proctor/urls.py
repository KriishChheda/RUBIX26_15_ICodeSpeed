from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    # path('auth/', include('apps.authentication.urls')),
    path('admin/', admin.site.urls),
    path('api/tests/', include('apps.tests.urls')),
    path('proctoring/', include('apps.proctoring.urls')),
    # path('results/', include('apps.results.urls')),
    path('api/auth/', include('apps.authentication.urls')),
    path('api/student/', include('apps.tests.student_urls')),
    path('api/results/', include('apps.tests.result_urls')),

]
