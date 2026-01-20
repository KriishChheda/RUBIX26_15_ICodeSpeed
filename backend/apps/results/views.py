from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from .models import TestResult
from .serializers import TestResultSerializer

class TestResultViewSet(viewsets.ModelViewSet):
    queryset = TestResult.objects.all()
    serializer_class = TestResultSerializer

    def list(self, request):
        results = self.queryset
        serializer = self.get_serializer(results, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        try:
            result = self.get_object()
            serializer = self.get_serializer(result)
            return Response(serializer.data)
        except TestResult.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None):
        try:
            result = self.get_object()
            serializer = self.get_serializer(result, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except TestResult.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

    def destroy(self, request, pk=None):
        try:
            result = self.get_object()
            result.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except TestResult.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)