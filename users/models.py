from django.db import models
from admins.models import modeldata

class DiagnosticResult(models.Model):
    user = models.ForeignKey(modeldata, on_delete=models.CASCADE)
    original_image = models.ImageField(upload_to='uploads/originals/')
    processed_image = models.CharField(max_length=255)  # Path to the detected image
    finding = models.CharField(max_length=100)          # Normal / Abnormal
    category = models.CharField(max_length=100, null=True, blank=True) # Complete/Incomplete/Dislocated
    confidence = models.FloatField(default=0.0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.finding} ({self.uploaded_at})"
