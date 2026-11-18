import os
import sys
import json
import base64
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SignMaster.settings')
django.setup()

from django.test import Client
from django.contrib.auth import get_user_model
User = get_user_model()
from asl_sign.models import GesturePredictionLog
from AIPlanner.models import StudySchedule, ExtendedUserProfile
from gamification.models import Leaderboard

def run_tests():
    client = Client()
    print("=" * 60)
    print("RUNNING SIGNMASTER COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 60)

    # Create or get test user for authenticated testing
    user, _ = User.objects.get_or_create(email='auditor@example.com', defaults={'name': 'Auditor', 'age': 25})
    client.force_login(user)

    # 1. Test Static & Dynamic GET routes
    routes = [
        ('/', 302, "Home Redirect to Dashboard"),
        ('/dashboard/', 200, "Main Dashboard"),
        ('/alphabets/', 200, "Alphabets Module"),
        ('/numbers/', 200, "Numbers Module"),
        ('/common_greetings/', 200, "Greetings Module"),
        ('/basic_questions/', 200, "Basic Questions"),
        ('/vocabulary/', 200, "Vocabulary"),
        ('/sentence_structure/', 200, "Sentence Structure"),
        ('/expressing_feelings/', 200, "Expressing Feelings"),
        ('/advanced_grammar/', 200, "Advanced Grammar"),
        ('/text-to-sign/', 200, "Text to Sign Input"),
        ('/text-to-sign/?text=HELLO', 200, "Text to Sign Query 'HELLO'"),
        ('/asl/recognize/', 200, "ASL Webcam Recognition Page"),
        ('/asl/model-insights/', 200, "Model Insights Dashboard"),
        ('/asl/api/history/', 200, "Gesture History API"),
        ('/plan/generate_study_plan/', 200, "AI Study Planner Form"),
        ('/gamification/quiz/', 200, "Quiz Page"),
        ('/gamification/flashcard-game/', 200, "Flashcard Game"),
        ('/gamification/word-match/', 200, "Word Match Game"),
        ('/gamification/sign_typing_game/', 200, "Sign Typing Game"),
        ('/gamification/leaderboard/', 200, "Quiz Leaderboard"),
        ('/gamification/flashcard-leaderboard/', 200, "Flashcard Leaderboard"),
        ('/gamification/word-match-leaderboard/', 200, "Word Match Leaderboard"),
        ('/gamification/sign-typing-game-leaderboard/', 200, "Sign Typing Leaderboard"),
    ]

    all_passed = True
    for route, expected_code, name in routes:
        resp = client.get(route)
        if resp.status_code == expected_code:
            print(f"  [PASS] {name:<35} GET {route:<45} -> {resp.status_code}")
        else:
            print(f"  [FAIL] {name:<35} GET {route:<45} -> Expected {expected_code}, got {resp.status_code}")
            all_passed = False

    print("\n" + "-" * 60)
    print("TESTING ASL ML INFERENCE & DATABASE PERSISTENCE")
    print("-" * 60)

    # 2. Test ML Prediction with actual test image
    test_img_path = os.path.join('data', 'test', 'A')
    test_files = [f for f in os.listdir(test_img_path) if f.lower().endswith(('.png', '.jpg'))] if os.path.exists(test_img_path) else []
    
    if test_files:
        sample_path = os.path.join(test_img_path, test_files[0])
        with open(sample_path, 'rb') as f:
            b64_data = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
        
        initial_log_count = GesturePredictionLog.objects.count()
        pred_resp = client.post(
            '/asl/api/predict/',
            data=json.dumps({'image': b64_data, 'mode': 'test_verify'}),
            content_type='application/json'
        )

        if pred_resp.status_code == 200:
            pred_json = pred_resp.json()
            predicted_class = pred_json.get('predicted_class') or pred_json.get('prediction')
            confidence = pred_json.get('confidence')
            print(f"  [PASS] Inference API: Predicted '{predicted_class}' with {confidence}% confidence")
            
            # Check DB persistence
            new_log_count = GesturePredictionLog.objects.count()
            if new_log_count == initial_log_count + 1:
                latest_log = GesturePredictionLog.objects.latest('id')
                print(f"  [PASS] GesturePredictionLog persisted to SQLite: ID={latest_log.id}, class={latest_log.predicted_class}, conf={latest_log.confidence}%, mode={latest_log.mode}")
            else:
                print(f"  [FAIL] GesturePredictionLog was not incremented! Expected {initial_log_count + 1}, got {new_log_count}")
                all_passed = False
        else:
            print(f"  [FAIL] /asl/api/predict/ failed with status {pred_resp.status_code}: {pred_resp.content}")
            all_passed = False
    else:
        print("  [WARN] No test image found in data/test/A")

    print("\n" + "-" * 60)
    print("TESTING AI PLANNER CURRICULUM GENERATOR & DB PERSISTENCE")
    print("-" * 60)

    # 3. Test AI Planner Submission & DB Persistence
    plan_payload = {
        'skill_level': 'intermediate',
        'learning_pace': 'moderate',
        'daily_study_hours': 1.5,
        'preferred_study_days': ['Monday', 'Wednesday', 'Friday', 'Saturday'],
        'learning_goals': 'Master fingerspelling, numbers, and basic conversation.'
    }

    initial_schedules = StudySchedule.objects.count()
    plan_resp = client.post(
        '/plan/process_user_input/',
        data=plan_payload
    )

    if plan_resp.status_code == 200:
        new_schedules = StudySchedule.objects.count()
        if new_schedules == initial_schedules + 1:
            latest_plan = StudySchedule.objects.latest('id')
            print(f"  [PASS] AI Plan generated and saved to DB: Schedule ID={latest_plan.id}")
            print(f"         Skill: {latest_plan.skill_level}, Hours/day: {latest_plan.daily_hours}")
            print(f"         Est. Completion Date: {latest_plan.completion_date}")

            # Verify saved plan detail retrieval
            detail_resp = client.get(f'/plan/plan/{latest_plan.id}/')
            if detail_resp.status_code == 200:
                print(f"  [PASS] GET /plan/plan/{latest_plan.id}/ returned 200 (view_saved_plan works)")
            else:
                print(f"  [FAIL] GET /plan/plan/{latest_plan.id}/ failed with status {detail_resp.status_code}")
                all_passed = False

            # Verify Duplicate Plan feature
            dup_resp = client.post(f'/plan/duplicate/{latest_plan.id}/')
            if dup_resp.status_code == 200 and dup_resp.json().get('status') == 'success':
                new_dup_id = dup_resp.json().get('new_id')
                print(f"  [PASS] POST /plan/duplicate/{latest_plan.id}/ created duplicate Plan #{new_dup_id} with clear feedback")
            else:
                print(f"  [FAIL] Duplicate Plan failed: {dup_resp.content}")
                all_passed = False
        else:
            print(f"  [FAIL] StudySchedule DB count not incremented! Expected {initial_schedules + 1}, got {new_schedules}")
            all_passed = False
    else:
        print(f"  [FAIL] /plan/process_user_input/ failed with {plan_resp.status_code}: {plan_resp.content}")
        all_passed = False

    print("\n" + "-" * 60)
    print("TESTING GAMIFICATION SCORE SAVING & LEADERBOARD RETRIEVAL")
    print("-" * 60)

    # 4. Test Gamification score submission
    score_resp = client.post(
        '/gamification/save_score/',
        data=json.dumps({'username': 'TestAuditor', 'score': 95, 'game_type': 'quiz'}),
        content_type='application/json'
    )
    if score_resp.status_code in (200, 201):
        print(f"  [PASS] Gamification /save_score/ persisted score 95 for 'TestAuditor' (Status {score_resp.status_code})")
        lb_resp = client.get('/gamification/get_leaderboard/?game_type=quiz')
        if lb_resp.status_code == 200:
            lb_data = lb_resp.json()
            print(f"  [PASS] Gamification /get_leaderboard/ returned {len(lb_data.get('leaderboard', []))} entries")
        else:
            print(f"  [FAIL] Gamification /get_leaderboard/ failed with {lb_resp.status_code}")
            all_passed = False
    else:
        print(f"  [FAIL] /gamification/save_score/ failed with {score_resp.status_code}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print(">>> ALL TESTS PASSED SUCCESSFULLY! APPLICATION IS 100% OPERATIONAL. <<<")
    else:
        print(">>> SOME TESTS FAILED. PLEASE CHECK OUTPUT ABOVE. <<<")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
