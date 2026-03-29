from rest_framework import serializers
from .models import DiagnosticResult
from admins.models import modeldata

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = modeldata
        fields = ['id', 'username', 'email', 'status']

class DiagnosticResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticResult
        fields = '__all__'
