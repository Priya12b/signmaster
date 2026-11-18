from django.urls import path
from . import views

urlpatterns = [
    path('input/', views.input_page, name='input_page'),
    path('text-to-sign/', views.text_to_sign, name='text_to_sign'),
    path('recognize/', views.asl_recognize_page, name='asl_recognize'),
    path('model-insights/', views.model_insights_page, name='model_insights'),
    path('api/predict/', views.api_predict_gesture, name='api_predict_gesture'),
    path('api/launch-webcam/', views.api_launch_webcam, name='api_launch_webcam'),
    path('api/history/', views.api_get_history, name='api_get_history'),
    path('api/clear-history/', views.api_clear_history, name='api_clear_history'),
]
