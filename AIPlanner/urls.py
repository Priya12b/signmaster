from django.urls import path
from . import views

app_name = 'AIPlanner'

urlpatterns = [
    path('process_user_input/', views.process_user_input, name='process_user_input'),
    path('generate-study-plan/', views.generate_study_plan_view, name='generate_study_plan'),
    path('generate_study_plan/', views.generate_study_plan_view, name='generate_study_plan_alt'),
    path('duplicate/<int:schedule_id>/', views.duplicate_study_plan, name='duplicate_study_plan'),
    path('plan/<int:schedule_id>/', views.view_saved_plan, name='view_saved_plan'),
]
