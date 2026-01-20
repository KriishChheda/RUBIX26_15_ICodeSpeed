# from django.utils.deprecation import MiddlewareMixin
# from django.http import JsonResponse
# from apps.authentication.utils import get_user_from_token

# class AuthMiddleware(MiddlewareMixin):
#     def process_request(self, request):
#         token = request.META.get('HTTP_AUTHORIZATION')
#         if not token:
#             return JsonResponse({'error': 'Unauthorized'}, status=401)

#         user = get_user_from_token(token)
#         if user is None:
#             return JsonResponse({'error': 'Invalid token'}, status=401)

#         request.user = user
#         return None