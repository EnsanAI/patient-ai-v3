#!/usr/bin/env python3
"""
Comprehensive Intent Testing Suite for Patient AI Service V2
Tests all appointment intents: booking, rescheduling, cancellation, checking, follow-up
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

class IntentTester:
    def __init__(self, base_url: str = "http://localhost:8000", session_id: str = "+971501234567"):
        self.base_url = base_url
        self.session_id = session_id
        self.test_results = []
        self.appointment_ids = []  # Store created appointment IDs for testing
        
    def send_message(self, message: str) -> Dict:
        """Send a message to the AI service"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json(),
                    "status_code": response.status_code
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 0
            }
    
    def run_test(self, test_name: str, message: str, expected_intent: str) -> Dict:
        """Run a single test"""
        print(f"\n{'='*80}")
        print(f"🧪 TEST: {test_name}")
        print(f"📝 Message: {message}")
        print(f"🎯 Expected Intent: {expected_intent}")
        print(f"{'='*80}")
        
        start_time = time.time()
        result = self.send_message(message)
        duration = time.time() - start_time
        
        test_result = {
            "test_name": test_name,
            "message": message,
            "expected_intent": expected_intent,
            "success": result["success"],
            "duration": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if result["success"]:
            data = result["data"]
            actual_intent = data.get("intent", "unknown")
            agent_name = data.get("agent_name", "unknown")
            response_text = data.get("response", "")
            
            intent_match = actual_intent == expected_intent
            
            test_result.update({
                "actual_intent": actual_intent,
                "agent_name": agent_name,
                "intent_match": intent_match,
                "response_preview": response_text[:200] if response_text else "No response",
                "full_response": response_text
            })
            
            # Extract appointment ID if present
            if "appointment_id" in data or "ID:" in response_text:
                # Try to extract ID from response
                import re
                id_match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', response_text)
                if id_match:
                    self.appointment_ids.append(id_match.group())
            
            status = "✅ PASS" if intent_match else "⚠️ INTENT MISMATCH"
            print(f"\n{status}")
            print(f"📌 Intent: {actual_intent}")
            print(f"🤖 Agent: {agent_name}")
            print(f"⏱️  Duration: {duration:.2f}s")
            print(f"💬 Response Preview: {response_text[:150]}...")
            
        else:
            test_result.update({
                "actual_intent": "error",
                "agent_name": "none",
                "intent_match": False,
                "error": result.get("error", "Unknown error")
            })
            
            print(f"\n❌ FAIL")
            print(f"❗ Error: {result.get('error', 'Unknown error')}")
        
        self.test_results.append(test_result)
        time.sleep(2)  # Brief pause between tests
        return test_result
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "="*80)
        print("🚀 PATIENT AI SERVICE V2 - COMPREHENSIVE INTENT TESTING")
        print("="*80)
        
        # Test 0: Registration (prerequisite)
        print("\n" + "="*80)
        print("📋 PREREQUISITE: Patient Registration")
        print("="*80)
        self.run_test(
            "Registration",
            "Hi, I'm a new patient. My name is Ahmed Hassan, phone +971501234567",
            "registration"
        )
        
        # Category 1: Appointment Booking
        print("\n" + "="*80)
        print("📅 CATEGORY 1: APPOINTMENT BOOKING TESTS")
        print("="*80)
        
        self.run_test(
            "Booking-1: Complete info",
            "I need an appointment with Dr. Mohammed Atef tomorrow at 3pm for a cleaning",
            "appointment_booking"
        )
        
        self.run_test(
            "Booking-2: Partial info",
            "I want to book a cleaning appointment",
            "appointment_booking"
        )
        
        # Category 2: Appointment Rescheduling
        print("\n" + "="*80)
        print("🔄 CATEGORY 2: APPOINTMENT RESCHEDULING TESTS")
        print("="*80)
        
        self.run_test(
            "Reschedule-1: Context-aware (it)",
            "can i move it to 3pm?",
            "appointment_reschedule"
        )
        
        self.run_test(
            "Reschedule-2: Explicit date/time",
            "reschedule my appointment to tomorrow at 11am",
            "appointment_reschedule"
        )
        
        self.run_test(
            "Reschedule-3: Change time only",
            "change my appointment time to 2pm",
            "appointment_reschedule"
        )
        
        self.run_test(
            "Reschedule-4: Generic request",
            "I need to reschedule",
            "appointment_reschedule"
        )
        
        # Category 3: Appointment Cancellation
        print("\n" + "="*80)
        print("❌ CATEGORY 3: APPOINTMENT CANCELLATION TESTS")
        print("="*80)
        
        self.run_test(
            "Cancel-1: Direct request",
            "cancel my appointment",
            "appointment_cancel"
        )
        
        self.run_test(
            "Cancel-2: Context-aware",
            "actually, i can't make it. cancel it",
            "appointment_cancel"
        )
        
        self.run_test(
            "Cancel-3: With reason",
            "cancel my appointment tomorrow, something came up",
            "appointment_cancel"
        )
        
        # Category 4: Appointment Checking
        print("\n" + "="*80)
        print("📋 CATEGORY 4: APPOINTMENT CHECKING TESTS")
        print("="*80)
        
        self.run_test(
            "Check-1: Next appointment",
            "when is my next appointment?",
            "appointment_check"
        )
        
        self.run_test(
            "Check-2: All appointments",
            "show all my appointments",
            "appointment_check"
        )
        
        self.run_test(
            "Check-3: Specific query",
            "what appointments do i have this week?",
            "appointment_check"
        )
        
        # Category 5: Follow-up
        print("\n" + "="*80)
        print("🏥 CATEGORY 5: FOLLOW-UP TESTS")
        print("="*80)
        
        self.run_test(
            "Followup-1: General request",
            "I need a follow-up appointment",
            "follow_up"
        )
        
        self.run_test(
            "Followup-2: Post-treatment question",
            "I had my cleaning yesterday, is sensitivity normal?",
            "follow_up"
        )
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS DASHBOARD")
        print("="*80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for t in self.test_results if t["success"])
        intent_matches = sum(1 for t in self.test_results if t.get("intent_match", False))
        failed_tests = total_tests - successful_tests
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        intent_accuracy = (intent_matches / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 OVERALL STATISTICS")
        print(f"{'─'*80}")
        print(f"Total Tests Run:        {total_tests}")
        print(f"✅ Successful:          {successful_tests} ({success_rate:.1f}%)")
        print(f"❌ Failed:              {failed_tests}")
        print(f"🎯 Intent Accuracy:     {intent_matches}/{total_tests} ({intent_accuracy:.1f}%)")
        print(f"⏱️  Avg Response Time:   {sum(t['duration'] for t in self.test_results) / total_tests:.2f}s")
        
        # Category breakdown
        categories = {
            "Booking": ["Booking"],
            "Rescheduling": ["Reschedule"],
            "Cancellation": ["Cancel"],
            "Checking": ["Check"],
            "Follow-up": ["Followup"]
        }
        
        print(f"\n📊 CATEGORY BREAKDOWN")
        print(f"{'─'*80}")
        
        for category, prefixes in categories.items():
            category_tests = [t for t in self.test_results if any(p in t["test_name"] for p in prefixes)]
            if category_tests:
                cat_total = len(category_tests)
                cat_success = sum(1 for t in category_tests if t.get("intent_match", False))
                cat_rate = (cat_success / cat_total * 100) if cat_total > 0 else 0
                status = "✅" if cat_rate >= 80 else "⚠️" if cat_rate >= 50 else "❌"
                print(f"{status} {category:15} {cat_success}/{cat_total} tests passed ({cat_rate:.0f}%)")
        
        # Detailed results
        print(f"\n📝 DETAILED TEST RESULTS")
        print(f"{'─'*80}")
        
        for i, test in enumerate(self.test_results, 1):
            status = "✅" if test.get("intent_match", False) else "❌" if test["success"] else "🔴"
            print(f"\n{i}. {status} {test['test_name']}")
            print(f"   Message: {test['message']}")
            print(f"   Expected Intent: {test['expected_intent']}")
            
            if test["success"]:
                print(f"   Actual Intent: {test.get('actual_intent', 'N/A')}")
                print(f"   Agent: {test.get('agent_name', 'N/A')}")
                print(f"   Duration: {test['duration']}s")
                if not test.get("intent_match", False):
                    print(f"   ⚠️  Intent mismatch detected")
            else:
                print(f"   Error: {test.get('error', 'Unknown error')}")
        
        # Save results to file
        report_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": success_rate,
                    "intent_accuracy": intent_accuracy
                },
                "tests": self.test_results
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"📄 Full results saved to: {report_file}")
        print(f"{'='*80}\n")
        
        # Final verdict
        if intent_accuracy >= 90:
            print("🎉 EXCELLENT! All intents working correctly!")
        elif intent_accuracy >= 70:
            print("✅ GOOD! Most intents working, minor issues detected.")
        elif intent_accuracy >= 50:
            print("⚠️  WARNING! Several intents not working correctly.")
        else:
            print("❌ CRITICAL! Major issues with intent handling.")

def main():
    """Main execution"""
    print("\n" + "🦷" * 40)
    print("PATIENT AI SERVICE V2 - INTENT TESTING SUITE")
    print("🦷" * 40 + "\n")
    
    tester = IntentTester()
    
    # Check service health
    try:
        response = requests.get(f"{tester.base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is healthy and ready for testing\n")
        else:
            print(f"⚠️  Service responded with status {response.status_code}\n")
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        print("Please ensure the service is running on http://localhost:8000")
        return
    
    # Run all tests
    tester.run_all_tests()

if __name__ == "__main__":
    main()

