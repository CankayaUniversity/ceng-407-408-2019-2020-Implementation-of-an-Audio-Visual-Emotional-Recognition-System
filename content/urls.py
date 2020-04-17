from django.urls import path
from . import views

urlpatterns = [
    path('add_content/', views.ContentAdd.as_view(), name='content_add'),
    path('list_contents/', views.content_list, name='contents_list'),
    path('content/<int:pk>/', views.content_detail, name='content_detail'),
    path('analyze_contents/', views.analyze_content, name='analyze_content'),
    path('emotion_analyze/', views.emotion_analyzes, name='emotion_analyze'),
    path('language_analyzes/', views.language_analyzes, name='language_analyzes'),
    path('test_visual/<int:pk>/', views.test_visual, name='test_visual'),
    path('test_audio/<int:pk>/', views.test_audio, name='test_audio'),
    path('test_fusion/<int:pk>/', views.test_fusion, name='test_fusion'),
    path('gender_analyzes/', views.gender_analyzes, name='gender_analyzes'),
]
