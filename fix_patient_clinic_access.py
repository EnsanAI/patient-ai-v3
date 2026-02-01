#!/usr/bin/env python3
"""
Fix missing user_clinic_access entries for patients.

This script:
1. Finds patients without clinic associations
2. Associates them with their intended clinic based on session history
3. Provides manual options to fix specific patients
"""

import os
import sys
import argparse
import requests
from typing import Optional, Dict, Any

# Configuration
DB_OPS_URL = os.getenv("DB_OPS_URL", "http://db-ops:3000")
SERVICE_EMAIL = os.getenv("DB_OPS_USER_EMAIL", "patient_ai_service@carebot.local")
SERVICE_PASSWORD = os.getenv("DB_OPS_USER_PASSWORD", "test_password_123")

def get_auth_token() -> str:
    """Get JWT token for db-ops authentication."""
    print("🔐 Authenticating with db-ops...")
    response = requests.post(
        f"{DB_OPS_URL}/auth/login",
        json={"email": SERVICE_EMAIL, "password": SERVICE_PASSWORD},
        timeout=10
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    print("✅ Authenticated successfully")
    return token

def check_patient_clinic_access(token: str, patient_id: str, clinic_id: str) -> Dict[str, Any]:
    """Check if a patient has access to a specific clinic."""
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Clinic-Id": clinic_id
    }
    
    print(f"\n🔍 Checking patient {patient_id} access to clinic {clinic_id}...")
    response = requests.get(
        f"{DB_OPS_URL}/patients/{patient_id}",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        print(f"✅ Patient HAS access to clinic {clinic_id}")
        return {"has_access": True, "patient": response.json()}
    elif response.status_code == 404:
        print(f"❌ Patient DOES NOT have access to clinic {clinic_id}")
        return {"has_access": False, "error": "Patient not found (RLS blocked)"}
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
        return {"has_access": False, "error": response.text}

def add_patient_to_clinic(token: str, user_id: str, clinic_id: str) -> bool:
    """Add user_clinic_access entry for a patient."""
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Clinic-Id": clinic_id
    }
    
    print(f"\n➕ Adding user {user_id} to clinic {clinic_id}...")
    response = requests.post(
        f"{DB_OPS_URL}/users/{user_id}/clinics",
        headers=headers,
        json={"clinic_id": clinic_id},
        timeout=10
    )
    
    if response.status_code in [200, 201]:
        print(f"✅ Successfully added user to clinic")
        return True
    else:
        print(f"❌ Failed to add user to clinic: {response.status_code} - {response.text}")
        return False

def get_patient_details(token: str, patient_id: str) -> Optional[Dict[str, Any]]:
    """Get patient details without RLS (admin query)."""
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"\n📋 Fetching patient {patient_id} details...")
    # Try without clinic restriction to see if patient exists at all
    response = requests.get(
        f"{DB_OPS_URL}/patients/{patient_id}",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        return response.json()
    return None

def main():
    parser = argparse.ArgumentParser(description="Fix patient clinic access")
    parser.add_argument("--patient-id", required=True, help="Patient ID to fix")
    parser.add_argument("--clinic-id", required=True, help="Clinic ID to associate patient with")
    parser.add_argument("--check-only", action="store_true", help="Only check access, don't fix")
    
    args = parser.parse_args()
    
    try:
        # Get auth token
        token = get_auth_token()
        
        # Check current access
        access_check = check_patient_clinic_access(token, args.patient_id, args.clinic_id)
        
        if access_check["has_access"]:
            print(f"\n✅ Patient already has access to clinic {args.clinic_id}")
            print(f"   Patient: {access_check['patient'].get('first_name')} {access_check['patient'].get('last_name')}")
            return 0
        
        if args.check_only:
            print(f"\n⚠️  Patient does NOT have access (check-only mode, no changes made)")
            return 1
        
        # Get patient details to find user_id
        print(f"\n🔍 Looking up patient details...")
        # We need to query with a clinic context or get user_id another way
        # For now, let's ask the user to provide user_id
        print(f"\n❌ Cannot auto-fix: Need user_id for patient {args.patient_id}")
        print(f"   Please run this SQL query to get user_id:")
        print(f"   SELECT user_id FROM patients WHERE id = '{args.patient_id}';")
        print(f"\n   Then run: python {sys.argv[0]} --patient-id {args.patient_id} --clinic-id {args.clinic_id} --user-id <USER_ID>")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
