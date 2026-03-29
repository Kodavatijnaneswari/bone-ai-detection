from django.db import models

# Create your models here.
class modeldata(models.Model):
    name=models.CharField(max_length=100)
    username=models.CharField(max_length=100,unique=True)
    password=models.CharField(max_length=100)
    mobile=models.CharField(max_length=100)
    email=models.EmailField(max_length=100)
    address=models.TextField(max_length=100)
    status=models.CharField(max_length=100,default='waiting')
    def __str__(self):
        return self.name