from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .import views

urlpatterns = [
	path('', views.index, name='index'),
    path('home', views.index, name='index'),
    path('clahe',views.clahe,name='clahe'),
    path('enhance',views.get_image,name='getimage'),
    path('rayleigh',views.rayleigh,name='rayleigh'),
    path('mip',views.mip,name='mip'),
    path('dcp',views.dcp,name='dcp'),
    path('rghs',views.rghs,name='rghs'),
    path('enhanceray',views.get_image_ray,name='getimageray'),
    path('enhanceRGHS',views.get_image_rghs,name='getimageRGHS'),
    path('restoremip',views.get_image_mip,name='getimagemip'),
    path('restoredcp',views.get_image_dcp,name='getimagedcp'),
    path('classify',views.classify,name='classify'),
    path('predict',views.classifyimage,name='classifyimage'),
    path('paper',views.paper,name='paper'),
    path('algorithms',views.algorithm,name='algorithm'),
    path('about-us',views.about,name='about'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)