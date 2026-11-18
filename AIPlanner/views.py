import os
import json
import datetime
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from .models import ExtendedUserProfile, StudySchedule, Course, Exercise

def generate_personalized_curriculum(skill_level, learning_speed, daily_hours, preferred_days):
    """
    Intelligent curriculum engine that dynamically generates a structured, multi-week
    ASL learning schedule customized to user level, speed, daily hours, and preferred days.
    """
    days = [d.capitalize() for d in preferred_days] if preferred_days else [
        "Monday", "Wednesday", "Friday"
    ]
    hours = max(1, int(daily_hours or 2))
    speed = (learning_speed or "Moderate").capitalize()
    level = (skill_level or "Beginner").capitalize()

    # Define curriculum modules by difficulty level
    if level == "Beginner":
        modules = [
            {
                "title": "Module 1: ASL Manual Alphabet Mastery (A-M)",
                "course": "ASL Alphabet & Finger-Spelling",
                "tasks": [
                    "Study hand shapes for letters A through G using SignMaster Alphabet module",
                    "Finger-spell your name and common words in front of mirror or webcam",
                    "Complete ASL Flashcard Game on SignMaster (Score at least 80%)",
                    "Practice letters H through M with steady hand orientation"
                ]
            },
            {
                "title": "Module 2: ASL Alphabet Completion (N-Z) & Numbers",
                "course": "ASL Alphabet & Numbers 0-9",
                "tasks": [
                    "Master letters N through T with accurate thumb positioning",
                    "Practice motion-sensitive letters (J and Z)",
                    "Learn ASL numbers 1-10 from the SignMaster Numbers module",
                    "Test your speed using the Sign Typing Game"
                ]
            },
            {
                "title": "Module 3: Essential Common Greetings & Polite Phrases",
                "course": "Common Greetings & Daily Phrases",
                "tasks": [
                    "Learn signs: Hello, Good Morning, Please, Thank You, Nice to meet you",
                    "Practice 2-way greeting dialogue scenarios",
                    "Take the SignMaster Beginner Quiz to test recall",
                    "Review tricky letters from previous weeks"
                ]
            },
            {
                "title": "Module 4: Real-Time Gesture Recognition & Evaluation",
                "course": "Interactive Vision Practice",
                "tasks": [
                    "Open SignMaster ASL Vision (Live Webcam) and test letters A, B, C, L, V, Y",
                    "Aim for >90% recognition confidence on the live webcam",
                    "Review Model Insights to understand classification metrics",
                    "Complete full 26-letter finger-spelling timed run"
                ]
            }
        ]
    elif level == "Intermediate":
        modules = [
            {
                "title": "Module 1: Everyday Vocabulary Expansion",
                "course": "Vocabulary & Everyday Expressions",
                "tasks": [
                    "Learn 25 core home and work vocabulary signs",
                    "Practice Match Word Game on SignMaster",
                    "Construct 5 simple Subject-Verb sign sentences",
                    "Drill high-speed finger-spelling for loan words"
                ]
            },
            {
                "title": "Module 2: Sentence Structure & Grammar Rules",
                "course": "ASL Syntax & Time-Topic-Comment",
                "tasks": [
                    "Study Topic-Comment sentence structure",
                    "Learn spatial reference and non-manual markers (NMM)",
                    "Practice Question-formation (Wh-questions with furrowed brows)",
                    "Complete Intermediate Quiz on SignMaster"
                ]
            },
            {
                "title": "Module 3: Expressing Feelings & Nuance",
                "course": "Feelings, Yes/No, & Facial Grammar",
                "tasks": [
                    "Learn signs for Happy, Sad, Excited, Confused, Eager",
                    "Coordinate facial expressions with sign intensity",
                    "Convert 10 English sentences to ASL using Text-to-Sign",
                    "Perform a 1-minute signed introduction"
                ]
            }
        ]
    else:  # Advanced
        modules = [
            {
                "title": "Module 1: Advanced Grammar & Classifiers",
                "course": "Classifiers & Spatial Grammar",
                "tasks": [
                    "Master CL:1, CL:3, and CL:V classifier predicates",
                    "Practice depicting spatial scenes and object movements",
                    "Complete Advanced Grammar module on SignMaster",
                    "Analyze signing flow and reduce hesitation pause"
                ]
            },
            {
                "title": "Module 2: Fluency & Speed Signing",
                "course": "Receptive & Expressive Fluency",
                "tasks": [
                    "Practice speed finger-spelling above 35 words per minute",
                    "Use SignMaster ASL Vision to verify rapid letter transitions",
                    "Compete on the Sign Typing Game Leaderboard",
                    "Conduct a 3-minute continuous sign language monologue"
                ]
            }
        ]

    # Build weekly schedule
    schedule_data = {}
    total_hours_est = 0
    today = datetime.date.today()

    for w_idx, mod in enumerate(modules, start=1):
        week_key = f"Week {w_idx}"
        schedule_data[week_key] = {
            "Course": mod["course"],
            "ModuleTitle": mod["title"],
            "Days": {}
        }
        
        # Distribute tasks among preferred days
        task_idx = 0
        for day in days:
            day_tasks = []
            if task_idx < len(mod["tasks"]):
                day_tasks.append(mod["tasks"][task_idx])
                task_idx += 1
            if task_idx < len(mod["tasks"]) and hours >= 2:
                day_tasks.append(mod["tasks"][task_idx])
                task_idx += 1
            if not day_tasks:
                day_tasks.append("Review previously learned signs and take a practice quiz")

            schedule_data[week_key]["Days"][day] = {
                "Hours": hours,
                "Tasks": day_tasks
            }
            total_hours_est += hours

    # Calculate expected completion date
    needed_sessions = len(modules) * len(days)
    curr_date = today
    sessions_counted = 0
    while sessions_counted < needed_sessions:
        curr_date += datetime.timedelta(days=1)
        if curr_date.strftime("%A") in days:
            sessions_counted += 1

    expected_completion = curr_date.strftime("%B %d, %Y")

    return {
        "status": "success",
        "data": {
            "skill_level": level,
            "learning_speed": speed,
            "daily_hours": hours,
            "preferred_days": days,
            "total_estimated_hours": total_hours_est,
            "expected_completion_date": expected_completion,
            "study_plan": schedule_data
        }
    }


@login_required(login_url='/login/')
def process_user_input(request):
    """Handles the study plan form view and plan generation."""
    user = request.user
    user_profile, _ = ExtendedUserProfile.objects.get_or_create(user=user)

    if request.method == "POST":
        skill_level = request.POST.get('skill_level', user_profile.skill_level or 'Beginner')
        learning_speed = request.POST.get('learning_speed', user_profile.learning_speed or 'Moderate')
        
        try:
            daily_hours = int(request.POST.get('daily_hours', 2))
        except (ValueError, TypeError):
            daily_hours = 2

        # Support both multi-select checkboxes or comma-separated text
        pref_raw = request.POST.get('preferred_days', '')
        if pref_raw:
            preferred_days = [d.strip() for d in pref_raw.split(',') if d.strip()]
        else:
            preferred_days = request.POST.getlist('preferred_days_list') or ['Monday', 'Wednesday', 'Friday']

        # Save to user profile
        user_profile.skill_level = skill_level
        user_profile.learning_speed = learning_speed
        user_profile.daily_hours = daily_hours
        user_profile.preferred_days = preferred_days
        user_profile.save()

        # Generate intelligent personalized plan
        curriculum_res = generate_personalized_curriculum(skill_level, learning_speed, daily_hours, preferred_days)
        plan_data = curriculum_res['data']

        # Save to database
        study_schedule = StudySchedule.objects.create(
            user=user,
            skill_level=skill_level,
            learning_speed=learning_speed,
            daily_hours=daily_hours,
            preferred_days=preferred_days,
            generated_study_plan=json.dumps(plan_data)
        )

        return render(request, "study_plan_result.html", {
            "study_plan": plan_data['study_plan'],
            "completion_date": plan_data['expected_completion_date'],
            "total_hours": plan_data['total_estimated_hours'],
            "skill_level": skill_level,
            "schedule_id": study_schedule.id
        })

    # GET request: Show form and list of previously generated plans for this user
    saved_plans = StudySchedule.objects.filter(user=user).order_by('-created_at')[:5]

    return render(request, "study_plan_form.html", {
        "user_profile": user_profile,
        "saved_plans": saved_plans
    })


@login_required(login_url='/login/')
def generate_study_plan_view(request):
    """API/form endpoint for study plan generation."""
    if request.method != "POST":
        return process_user_input(request)
    user = request.user
    user_profile, _ = ExtendedUserProfile.objects.get_or_create(user=user)

    try:
        body = json.loads(request.body.decode('utf-8'))
    except Exception:
        body = {}

    skill_level = body.get('skill_level', 'Beginner')
    learning_speed = body.get('learning_speed', 'Moderate')
    try:
        daily_hours = int(body.get('daily_hours', 2))
    except (ValueError, TypeError):
        daily_hours = 2

    pref_raw = body.get('preferred_days', 'Monday, Wednesday, Friday')
    preferred_days = [d.strip() for d in pref_raw.split(',') if d.strip()]

    user_profile.skill_level = skill_level
    user_profile.learning_speed = learning_speed
    user_profile.daily_hours = daily_hours
    user_profile.preferred_days = preferred_days
    user_profile.save()

    curriculum_res = generate_personalized_curriculum(skill_level, learning_speed, daily_hours, preferred_days)
    plan_data = curriculum_res['data']

    StudySchedule.objects.create(
        user=user,
        skill_level=skill_level,
        learning_speed=learning_speed,
        daily_hours=daily_hours,
        preferred_days=preferred_days,
        generated_study_plan=json.dumps(plan_data)
    )

    return JsonResponse({"study_plan": curriculum_res})


@login_required(login_url='/login/')
def duplicate_study_plan(request, schedule_id):
    """Duplicates an existing study schedule with a new ID, timestamp, and clear feedback."""
    try:
        orig = StudySchedule.objects.get(id=schedule_id, user=request.user)
    except StudySchedule.DoesNotExist:
        orig = StudySchedule.objects.filter(id=schedule_id).first()
        if not orig:
            return JsonResponse({'status': 'error', 'message': 'Study plan not found'}, status=404)

    new_plan = StudySchedule.objects.create(
        user=request.user,
        skill_level=orig.skill_level,
        learning_speed=orig.learning_speed,
        daily_hours=orig.daily_hours,
        preferred_days=orig.preferred_days,
        generated_study_plan=orig.generated_study_plan,
        completion_date=orig.completion_date,
        progress_percentage=0
    )

    return JsonResponse({
        'status': 'success',
        'message': f'Study plan #{schedule_id} successfully duplicated as #{new_plan.id}!',
        'new_id': new_plan.id,
        'redirect_url': f'/plan/plan/{new_plan.id}/'
    })


@login_required(login_url='/login/')
def view_saved_plan(request, schedule_id):
    """View a previously saved study schedule."""
    try:
        schedule = StudySchedule.objects.get(id=schedule_id, user=request.user)
        plan_dict = json.loads(schedule.generated_study_plan)
        return render(request, "study_plan_result.html", {
            "study_plan": plan_dict.get('study_plan', {}),
            "completion_date": plan_dict.get('expected_completion_date', schedule.completion_date),
            "total_hours": plan_dict.get('total_estimated_hours', 20),
            "skill_level": schedule.skill_level,
            "schedule_id": schedule.id
        })
    except Exception:
        return redirect('AIPlanner:process_user_input')
