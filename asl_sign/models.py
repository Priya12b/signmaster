from django.db import models
from django.conf import settings

class ASLSign(models.Model):
    character = models.CharField(max_length=1, unique=True)  # A-Z, 0-9, space
    image_path = models.CharField(max_length=255)  # Path to image file

    def __str__(self):
        return self.character

class GesturePredictionLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    predicted_class = models.CharField(max_length=10)
    confidence = models.FloatField()
    mode = models.CharField(max_length=20, default='webcam')  # 'webcam' or 'upload'
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence*100:.1f}%) at {self.created_at}"
