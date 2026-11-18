import datetime
from django.db import models
from django.contrib.auth import get_user_model
from Dashboard.models import CustomUser

User = get_user_model()

# Extended User Profile Model
class ExtendedUserProfile(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
    skill_level = models.CharField(
        max_length=20, 
        choices=[('Beginner', 'Beginner'), ('Intermediate', 'Intermediate'), ('Advanced', 'Advanced')],
        default='Beginner'
    )
    learning_speed = models.CharField(
        max_length=10, 
        choices=[('Slow', 'Slow'), ('Moderate', 'Moderate'), ('Fast', 'Fast')],
        default='Moderate'
    )
    daily_hours = models.IntegerField(default=2)
    preferred_days = models.JSONField(default=list)  # e.g. ['Monday', 'Wednesday', 'Friday']

    def __str__(self):
        return f"{self.user.name or self.user.email}'s Profile"
    
class Course(models.Model):
    name = models.CharField(max_length=255)
    difficulty_level = models.CharField(max_length=20, choices=[('Beginner', 'Beginner'), ('Intermediate', 'Intermediate'), ('Advanced', 'Advanced')])
    description = models.TextField()
    topics_and_hours = models.JSONField(default=dict)  # Format: { "Topic Name": hours }
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
class Exercise(models.Model):
    course = models.ForeignKey(Course, related_name='exercises', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    description = models.TextField()
    difficulty_level = models.CharField(max_length=20, choices=[('Beginner', 'Beginner'), ('Intermediate', 'Intermediate'), ('Advanced', 'Advanced')])
    exercises_and_hours = models.JSONField(default=dict)
    
    def __str__(self):
        return self.name

# Study Schedule Model
class StudySchedule(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    skill_level = models.CharField(max_length=20)
    learning_speed = models.CharField(max_length=10)
    daily_hours = models.IntegerField(default=2)
    preferred_days = models.JSONField(default=list)
    generated_study_plan = models.TextField(null=True, blank=True)
    completion_date = models.DateField(null=True, blank=True)
    progress_percentage = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.email}'s Study Schedule ({self.skill_level})"
    
    def calculate_completion_date(self, total_hours=None):
        if not total_hours:
            hours_map = {'Beginner': 16, 'Intermediate': 28, 'Advanced': 42}
            total_hours = hours_map.get(self.skill_level, 20)

        daily_hours = max(1, self.daily_hours or 2)
        days_needed = max(1, int(total_hours / daily_hours))
        today = datetime.date.today()
        
        pref_days = [d.capitalize() for d in self.preferred_days] if self.preferred_days else [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
        ]
        
        current = today
        count = 0
        max_loop = 365
        loop = 0
        
        while count < days_needed and loop < max_loop:
            current += datetime.timedelta(days=1)
            loop += 1
            if current.strftime('%A') in pref_days:
                count += 1
        
        return current

    def save(self, *args, **kwargs):
        if not self.completion_date:
            self.completion_date = self.calculate_completion_date()
        super().save(*args, **kwargs)
