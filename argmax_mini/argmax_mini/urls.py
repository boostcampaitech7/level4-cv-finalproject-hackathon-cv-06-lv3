from django.contrib import admin
from django.urls import path, include
from rest_framework import permissions
from rest_framework.authentication import SessionAuthentication
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


# UI 연결 후 삭제
class CustomSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return


schema_view = get_schema_view(
    openapi.Info(
        title="argmax mini API",
        default_version="v1",
        description="API documentation for argmax_mini project",
    ),
    public=True,
    authentication_classes=(CustomSessionAuthentication,),
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
    path(
        "swagger/", schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui"),
    path(
        "redoc/", schema_view.with_ui("redoc", cache_timeout=0),
        name="schema-redoc"),
    path('data-processing/', include('data_processing.urls')),
]
