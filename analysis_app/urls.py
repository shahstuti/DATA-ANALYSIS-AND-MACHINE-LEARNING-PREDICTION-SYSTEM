# from django.urls import path
# from . import views

# app_name = 'analysis_app'

# urlpatterns = [
#     path('', views.upload_csv, name='upload_csv'),
#     path('analysis/', views.analysis, name='analysis'),
#     path('cleaning/', views.cleaning, name='cleaning'),
#     path('clean_data/', views.clean_data, name='clean_data'),
#     path('graphs/', views.graphs, name='graphs'),
#     path('generate_chart/', views.generate_chart, name='generate_chart'),
#     path('prediction/', views.prediction, name='prediction'),
#     path('run_prediction/', views.run_prediction, name='run_prediction'),
#     path('model_comparison/', views.model_comparison, name='model_comparison'),
# ]

from django.urls import path
from . import views

app_name = 'analysis_app'

urlpatterns = [
    # HTML endpoints (for direct user access)
    path('', views.upload_csv, name='upload_csv'),
    path('upload.html', views.upload_csv, name='upload_html'),
    path('analysis.html', views.analysis, name='analysis_html'),
    path('cleaning.html', views.cleaning, name='cleaning_html'),
    path('graphs.html', views.graphs, name='graphs_html'),
    path('prediction.html', views.prediction, name='prediction_html'),
    path('model_comparison.html', views.model_comparison, name='model_comparison_html'),

    # UUID endpoints (for API access and dynamic loading)
    path('analysis/<uuid:dataset_id>/', views.analysis, name='analysis'),
    path('cleaning/<uuid:dataset_id>/', views.cleaning, name='cleaning'),
    path('graphs/<uuid:dataset_id>/', views.graphs, name='graphs'),
    path('prediction/<uuid:dataset_id>/', views.prediction, name='prediction'),
    path('model_comparison/<uuid:dataset_id>/', views.model_comparison, name='model_comparison'),

    # API endpoints
    path('clean_data/<uuid:dataset_id>/', views.clean_data, name='clean_data'),
    path('generate_chart/<uuid:dataset_id>/', views.generate_chart, name='generate_chart'),
    path('run_prediction/<uuid:dataset_id>/', views.run_prediction, name='run_prediction')
]