import requests
import os
import logging
import time
import json
from typing import Optional, Dict, Any, List # Added for type hinting

# Configure logging
logger = logging.getLogger(__name__)

class DbOpsClient:
    def __init__(self, base_url: Optional[str] = None, user_email: Optional[str] = None, user_password: Optional[str] = None):
        self.base_url = base_url or os.environ.get("DB_OPS_URL")
        if not self.base_url:
            # For Docker: use db-ops service name on port 3000 (internal), for local: use localhost:8001
            if os.environ.get("DOCKER_HOST") or os.environ.get("HOSTNAME", "").startswith("carebot"):
                logger.info("Running in Docker environment, using db-ops service hostname")
                self.base_url = "http://db-ops:3000"
            else:
                logger.warning("DB_OPS_URL not set. For Docker, use db-ops:3000. For local, use localhost:8001")
                # Default to localhost for local development
                self.base_url = "http://localhost:8001"

        logger.info(f"DbOpsClient initialized with base_url: {self.base_url}")
        
        self.user_email = user_email or os.environ.get("DB_OPS_USER_EMAIL")
        self.user_password = user_password or os.environ.get("DB_OPS_USER_PASSWORD")
        self.auth_token: Optional[str] = os.environ.get("PATIENT_AI_SERVICE_AUTH_TOKEN",None)
        self.refresh_token: Optional[str] = os.environ.get("PATIENT_AI_SERVICE_REFRESH_TOKEN",None)
        self.token_expiry: Optional[float] = None
        self.max_auth_attempts = 3
        self.auth_attempt_count = 0
        if not self.user_email or not self.user_password:
            logger.error("DB_OPS_USER_EMAIL or DB_OPS_USER_PASSWORD environment variables not set nor provided. Cannot authenticate.")
        else:
            self._get_auth_token() # Initial attempt to get token upon instantiation

    def _get_auth_token(self) -> Optional[str]:
        """
        Get a valid auth token through the following strategy:
        1. Validate existing token (if present)
        2. Refresh token if needed
        3. Login with email/password
        4. Return token if successful, None otherwise

        This is the version that doesn't try registration.
        """

        logger.info(f"🔑 Getting auth token for {self.user_email}")

        # Strategy 1: Validate existing token
        if self.auth_token:
            if self._validate_auth_token():
                logger.info(f"✅ Using existing valid token")
                return self.auth_token
            else:
                logger.warning(f"⚠️ Existing token is invalid")

        # Strategy 2: Try to refresh token
        if self.refresh_token and self.auth_token:
            if self._refresh_access_token():
                return self.auth_token

        # Strategy 3: Login with email/password
        if self.user_email and self.user_password:
            if self._login():
                return self.auth_token

        # All strategies failed
        logger.error(f"❌ Could not obtain auth token - all strategies failed")
        logger.error(f"   - No valid cached token")
        logger.error(f"   - Could not refresh token")
        logger.error(f"   - Could not login")

        return None
    
    def _login(self) -> bool:
        """
        Attempt to login with email/password.
        Returns True if successful, False otherwise.
        """
        if not self.user_email or not self.user_password:
            logger.error(f"❌ Cannot login: missing email/password")
            return False

        try:
            logger.info(f"🔐 Attempting login as {self.user_email}")

            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"email": self.user_email, "password": self.user_password},
                timeout=10
            )

            # Accept both 200 (OK) and 201 (Created) as success
            if response.status_code in [200, 201]:
                data = response.json()
                self.auth_token = data.get("accessToken")
                self.refresh_token = data.get("refreshToken")

                logger.info(f"✅ Login successful for {self.user_email}")
                return True

            elif response.status_code == 401:
                logger.error(f"❌ Login failed: Invalid credentials")
                return False

            else:
                logger.error(f"❌ Login failed with {response.status_code}: {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.error(f"❌ Login timeout")
            return False

        except Exception as e:
            logger.error(f"❌ Error during login: {e}")
            return False
            
    def _refresh_access_token(self) -> bool:
        """
        Attempt to refresh the access token using refresh token.
        Returns True if successful, False otherwise.
        """
        if not self.refresh_token:
            logger.warning(f"⚠️ No refresh token available")
            return False

        try:
            logger.info(f"🔄 Attempting token refresh...")

            response = requests.post(
                f"{self.base_url}/auth/refresh",
                json={"refreshToken": self.refresh_token},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get("accessToken")
                self.refresh_token = data.get("refreshToken", self.refresh_token)

                logger.info(f"✅ Token refreshed successfully")
                return True
            elif response.status_code == 401:
                logger.warning(f"⚠️ Refresh token invalid (401), need to re-login")
                self.refresh_token = None
                return False
            else:
                logger.warning(f"⚠️ Token refresh failed with {response.status_code}: {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Token refresh timeout")
            return False

        except Exception as e:
            logger.error(f"❌ Error refreshing token: {e}")
            return False

    def _validate_auth_token(self) -> bool:
        """
        Validate that the current auth token is still valid.
        Returns True if valid, False if not.
        """
        if not self.auth_token:
            return False

        try:
            response = requests.post(
                f"{self.base_url}/auth/validate",
                json={"accessToken": self.auth_token},
                timeout=5  # Quick timeout for validation
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    logger.debug(f"✅ Auth token validated successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Auth token validation failed: {data.get('reason', 'unknown')}")
                    return False
            else:
                logger.warning(f"⚠️ Token validation returned {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Token validation timeout - assuming valid")
            return True  # Optimistic - assume valid on timeout

        except Exception as e:
            logger.warning(f"⚠️ Error validating token: {e}")
            return False

    def _should_refresh_token(self) -> bool:
        """Check if token needs refresh based on expiry."""
        if not self.auth_token:
            return True

        if not self.token_expiry:
            return False  # No expiry info, assume valid

        # Refresh if token expires within 60 seconds
        return time.time() > (self.token_expiry - 60)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        attempts: int = 2
    ) -> Optional[dict]:
        """
        Make an authenticated HTTP request with automatic token refresh.

        IMPROVEMENTS:
        - Only retries on 401 (unauthorized)
        - Better error messages
        - Timeout handling
        - Request validation
        """

        # Ensure we have a valid token before making the request
        if not self.auth_token and not endpoint.startswith("/auth"):
            logger.warning(f"⚠️ No auth token for {method} {endpoint}, attempting to get one...")
            if not self._get_auth_token():
                logger.error(f"❌ Could not obtain auth token")
                return None

        url = f"{self.base_url}{endpoint}"
        headers = {}

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        logger.debug(f"📤 {method} {endpoint}")

        for attempt in range(attempts):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=15
                )

                # Success cases
                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(f"📥 Response (200): {response_data}")
                    return response_data

                elif response.status_code == 201:
                    response_data = response.json()
                    logger.info(f"📥 Response (201 Created): {response_data}")
                    return response_data

                elif response.status_code == 204:
                    return True  # No content success

                # 401 - Try to refresh token and retry
                elif response.status_code == 401 and attempt < attempts - 1:
                    logger.warning(f"⚠️ Got 401 (unauthorized), attempting refresh...")

                    if self._refresh_access_token():
                        logger.info(f"✅ Token refreshed, retrying request")
                        headers["Authorization"] = f"Bearer {self.auth_token}"
                        continue  # Retry the request
                    else:
                        logger.error(f"❌ Could not refresh token")
                        return None

                # Other errors
                else:
                    error_text = response.text[:200] if response.text else "No details"
                    logger.error(
                        f"❌ {method} {endpoint} failed with {response.status_code}: {error_text}"
                    )
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"❌ Request timeout (attempt {attempt + 1}/{attempts})")
                if attempt < attempts - 1:
                    time.sleep(1)  # Wait before retry
                    continue

            except requests.exceptions.ConnectionError as e:
                logger.error(f"❌ Connection error: {e}")
                return None

            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}", exc_info=True)
                return None

        logger.error(f"❌ Request failed after {attempts} attempts")
        return None

    # --- Specific API call functions ---

    def get_doctors(self, language: Optional[str] = None, dialect: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Fetching doctors. Language: {language}, Dialect: {dialect}")
        if language:
            return self._make_request("GET", f"/doctors/language/{language}")
        elif dialect:
            return self._make_request("GET", f"/doctors/dialect/{dialect}")
        else:
            return self._make_request("GET", "/doctors")

    def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Fetching doctor by ID: {doctor_id}")
        return self._make_request("GET", f"/doctors/{doctor_id}")

    def get_specialties(self) -> Optional[List[Dict[str, Any]]]:
        logger.info("Fetching all specialties.")
        return self._make_request("GET", "/specialties")

    def get_doctor_availability(self, doctor_id: str, date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Fetching availability for doctor ID: {doctor_id}, Date: {date}")
        params = {}
        if date:
            params["date"] = date
        availability = self._make_request("GET", f"/doctors/{doctor_id}/availability", params=params)
        logger.info(f"Availability: {availability}")
        return availability

    def get_doctor_appointments(self, doctor_id: str, date: Optional[str] = None, status: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get existing appointments for a doctor, optionally filtered by date and status.
        
        Args:
            doctor_id: Doctor's UUID
            date: Optional date in YYYY-MM-DD format to filter appointments
            status: Optional status filter (e.g., 'scheduled', 'confirmed', 'cancelled')
        
        Returns:
            List of appointment dictionaries or None if error
        """
        logger.info(f"Fetching appointments for doctor ID: {doctor_id}, Date: {date}, Status: {status}")
        try:
            # Try to get appointments via doctor-specific endpoint if available
            params = {}
            if date:
                params["date"] = date
            if status:
                params["status"] = status
            
            # Try /doctors/{id}/appointments first, fallback to /appointments with filters
            appointments = self._make_request("GET", f"/doctors/{doctor_id}/appointments", params=params)
            if appointments is not None:
                return appointments
            
            # Fallback: get all appointments and filter locally
            all_appointments = self._make_request("GET", "/appointments", params={})
            if not all_appointments:
                return []
            
            # Filter by doctor_id
            filtered = [apt for apt in all_appointments if apt.get("doctor_id") == doctor_id]
            
            # Filter by date if provided
            if date:
                filtered = [apt for apt in filtered if apt.get("appointment_date") == date]
            
            # Filter by status if provided
            if status:
                filtered = [apt for apt in filtered if apt.get("status") == status]
            
            logger.info(f"Found {len(filtered)} appointments for doctor {doctor_id}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error fetching doctor appointments: {e}")
            return []

    def get_available_doctors(self, date: str, start_time: str, end_time: str, language: Optional[str] = None, dialect: Optional[str] = None, specialty_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        logger.info(f"Fetching available doctors. Date: {date}, Start: {start_time}, End: {end_time}, Lang: {language}, Dialect: {dialect}, Specialty: {specialty_id}")
        params = {
            "date": date,
            "startTime": start_time,
            "endTime": end_time
        }
        if language:
            params["language"] = language
        if dialect:
            params["dialect"] = dialect
        
        endpoint = "/doctors/available"
        if specialty_id:
            endpoint = f"/doctors/available/specialty/{specialty_id}"
            
        return self._make_request("GET", endpoint, params=params)

    def get_available_time_slots(
        self, 
        doctor_id: str, 
        date: str, 
        slot_duration_minutes: int = 30
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get actual generated available time slots for a doctor on a specific date.
        
        This returns REAL timeslots that:
        - Are generated from doctor's availability windows
        - Are filtered to exclude booked appointments
        - Are within working hours
        - Respect appointment duration rules
        
        Args:
            doctor_id: Doctor UUID
            date: Date in YYYY-MM-DD format
            slot_duration_minutes: Duration of each slot (default 30)
        
        Returns:
            List of timeslot dicts with start_time, end_time, is_available
            Returns None if error, empty list if no slots available
        """
        logger.info(f"Fetching available time slots for doctor {doctor_id} on {date} (duration: {slot_duration_minutes}min)")
        
        params = {
            "date": date,
            "slot_duration": slot_duration_minutes
        }
        
        # Call GET /appointments/:doctorId/time-slots?date=YYYY-MM-DD&slot_duration=30
        result = self._make_request("GET", f"/appointments/{doctor_id}/time-slots", params=params)
        
        if result and isinstance(result, dict):
            # Response format: {"success": true, "data": {"slots": [...]}}
            if result.get("success") and result.get("data"):
                slots = result.get("data", {}).get("slots", [])
                logger.info(f"Found {len(slots)} available time slots for doctor {doctor_id} on {date}")
                return slots
            else:
                logger.warning(f"No slots in response for doctor {doctor_id} on {date}")
                return []
        elif result and isinstance(result, list):
            # Direct list response
            logger.info(f"Found {len(result)} available time slots for doctor {doctor_id} on {date}")
            return result
        else:
            logger.warning(f"No available time slots found for doctor {doctor_id} on {date}")
            return []

    def create_appointment(self, clinic_id: str, patient_id: str, doctor_id: str, appointment_type_id: str, appointment_date: str, start_time: str, end_time: str, status: Optional[str] = None, reason: Optional[str] = None, emergency_level: Optional[str] = None, procedure_id: Optional[str] = None, notes: Optional[str] = None, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.info(f"Creating appointment for patient {patient_id} with doctor {doctor_id}")
        if request_id:
            logger.info(f"Request ID for idempotency: {request_id}")

        payload = {
            "clinic_id": clinic_id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "appointment_type_id": appointment_type_id,
            "appointment_date": appointment_date, 
            "start_time": start_time, 
            "end_time": end_time, 
            "reason": reason,
            "emergency_level": emergency_level or "routine",
            "procedure_id": procedure_id,
            "notes": notes,
            "request_id": request_id  # Add request_id for idempotency
        }
        # Note: status is not in CreateAppointmentDto - it's set automatically by the API to 'scheduled'
        logger.info(f"Payload for creating the appointment: {payload}")
        result = self._make_request("POST", "/appointments", json_data=payload)
        if result:
            appointment_id = result.get('id')
            logger.info(f"✅ Appointment created successfully: {appointment_id}")
            logger.debug(f"📋 Full appointment response: {result}")
            if not appointment_id:
                logger.warning(f"⚠️ Appointment response received but 'id' field is missing. Response keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        else:
            logger.error(f"❌ Failed to create appointment - API returned None or error")
        return result
    
    def get_appointment_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get appointment by request_id for idempotency check."""
        logger.info(f"Checking for existing appointment with request_id: {request_id}")
        try:
            # Try to get via dedicated endpoint if available
            result = self._make_request("GET", f"/appointments/by-request-id/{request_id}")
            if result:
                logger.info(f"✅ Found existing appointment: {result.get('id')}")
                return result
            
            # Fallback: get all appointments and filter
            all_appointments = self._make_request("GET", "/appointments")
            if all_appointments and isinstance(all_appointments, list):
                for apt in all_appointments:
                    if apt.get("request_id") == request_id:
                        logger.info(f"✅ Found existing appointment via fallback: {apt.get('id')}")
                        return apt
            
            logger.info(f"No existing appointment found with request_id: {request_id}")
            return None
        except Exception as e:
            logger.error(f"Error checking for existing appointment: {e}")
            return None
    
    def create_audit_log(self, entity_type: str, entity_id: str, action: str, correlation_id: str, request_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create an audit log entry."""
        logger.info(f"Creating audit log: {entity_type}.{action} - {entity_id}")
        payload = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "correlation_id": correlation_id,
            "request_id": request_id,
            "metadata": metadata,
            "timestamp": time.time()
        }
        try:
            result = self._make_request("POST", "/audit-logs", json_data=payload)
            if result:
                logger.info(f"✅ Audit log created successfully")
            return result
        except Exception as e:
            logger.warning(f"⚠️ Failed to create audit log: {e}")
            return None

    def get_appointment_by_id(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Fetching appointment by ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/{appointment_id}")

    def cancel_appointment(self, appointment_id: str, cancellation_reason: str) -> Optional[Any]: # Can return True for 204 or Dict
        logger.info(f"Cancelling appointment ID: {appointment_id}")
        payload = {"cancellation_reason": cancellation_reason}
        return self._make_request("PATCH", f"/appointments/{appointment_id}/cancel", json_data=payload)

    def reschedule_appointment(self, appointment_id: str, appointment_date: str, start_time: str, end_time: str, reschedule_reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        logger.info(f"Rescheduling appointment ID: {appointment_id}")
        payload = {
            "appointment_date": appointment_date,
            "start_time": start_time,
            "end_time": end_time
        }
        return self._make_request("PATCH", f"/appointments/{appointment_id}/reschedule", json_data=payload)

    def update_appointment(self, appointment_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an appointment with one or more parameters.
        
        This is a flexible method that can update any combination of appointment fields.
        Supported fields (all optional):
        - doctor_id: Change the doctor
        - clinic_id: Change the clinic
        - patient_id: Change the patient
        - appointment_type_id: Change appointment type
        - appointment_date: Change the date (YYYY-MM-DD)
        - start_time: Change start time (HH:MM)
        - end_time: Change end time (HH:MM)
        - status: Change status (scheduled, confirmed, checked_in, in_progress, completed, cancelled, no_show, rescheduled)
        - reason: Update reason
        - notes: Update notes
        - emergency_level: Change emergency level (routine, urgent, emergency, critical)
        - follow_up_required: Boolean
        - follow_up_days: Number of days
        - procedure_id: Procedure ID (UUID)
        
        Args:
            appointment_id: The appointment ID to update
            updates: Dictionary of fields to update (only include fields you want to change)
        
        Returns:
            Updated appointment dict if successful, None if error
        """
        logger.info(f"Updating appointment ID: {appointment_id} with fields: {list(updates.keys())}")
        
        # Phase 1: Enhanced logging - Log exact payload being sent
        logger.info(f"🔍 PATCH /appointments/{appointment_id} payload: {json.dumps(updates, indent=2)}")
        
        # Track if doctor_id is being updated for verification
        doctor_id_in_request = updates.get("doctor_id")
        if doctor_id_in_request:
            logger.info(f"🔍 Attempting to update doctor_id to: {doctor_id_in_request}")
        
        # Make the API request
        result = self._make_request("PATCH", f"/appointments/{appointment_id}", json_data=updates)
        
        # Phase 1: Response validation - Verify persistence by fetching appointment
        # Phase 4: Ensure response reflects actual DB state
        if result:
            logger.info(f"🔍 API response received. Verifying persistence and fetching actual DB state...")
            logger.info(f"🔍 Response doctor_id: {result.get('doctor_id')}")
            
            # Fetch appointment from DB to get actual state (not just API response echo)
            try:
                fetched_appointment = self.get_appointment_by_id(appointment_id)
                if fetched_appointment:
                    # Phase 4: Return fetched appointment to ensure response reflects DB state
                    logger.info(f"🔍 Returning fetched appointment (actual DB state) instead of API response")
                    
                    # Verify doctor_id if it was in the request
                    if doctor_id_in_request:
                        fetched_doctor_id = fetched_appointment.get("doctor_id")
                        logger.info(f"🔍 Fetched appointment doctor_id from DB: {fetched_doctor_id}")
                        
                        # Compare response vs. database state
                        response_doctor_id = result.get("doctor_id")
                        if response_doctor_id != fetched_doctor_id:
                            logger.warning(
                                f"⚠️ DOCTOR_ID MISMATCH DETECTED: "
                                f"Requested={doctor_id_in_request}, "
                                f"Response={response_doctor_id}, "
                                f"Database={fetched_doctor_id}"
                            )
                        elif fetched_doctor_id != doctor_id_in_request:
                            logger.error(
                                f"❌ DOCTOR_ID UPDATE FAILED: "
                                f"Requested={doctor_id_in_request}, "
                                f"Database still has={fetched_doctor_id}. "
                                f"Backend may not support doctor_id updates."
                            )
                        else:
                            logger.info(f"✅ Doctor_id successfully persisted: {fetched_doctor_id}")
                    
                    # Return fetched appointment (actual DB state) instead of API response
                    return fetched_appointment
                else:
                    logger.warning(f"⚠️ Could not fetch appointment {appointment_id} for verification, returning API response")
            except Exception as e:
                logger.warning(f"⚠️ Error fetching appointment for verification: {e}, returning API response")
        
        return result

    def get_patient_by_id(self, patient_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not patient_id:
            logger.warning(f"DbOpsClient.get_patient_by_id called with missing or invalid patient_id: '{patient_id}'. Aborting API call.")
            return None
        logger.info(f"Fetching patient by ID: {patient_id}")
        return self._make_request("GET", f"/patients/{patient_id}")
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        logger.warning("get_user_by_email: This method currently fetches all users and filters by email locally, which can be inefficient. Consider a dedicated API endpoint in db-ops for direct email lookup if performance is critical.")
        users = self._make_request("GET", "/users")
        if users and isinstance(users, list):
            for user in users:
                if user.get("email") == email:
                    return user
        return None

    def get_patient_by_phone_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Gets patient information using their phone number - simplified international version.
        
        Args:
            phone_number: The patient's phone number (any international format)
            
        Returns:
            Dict with patient data if found, None if not found.
        """
        logger.info(f"Fetching patient by phone number: {phone_number}")
        
        # Clean the phone number
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        
        # Build phone number variations
        variations = [
            phone_number,           # Original
            clean_phone,           # Digits only
            f"+{clean_phone}",     # With +
        ]
        
        # Handle leading zero (national format)
        if clean_phone.startswith('0') and len(clean_phone) > 6:
            without_zero = clean_phone[1:]
            variations.extend([without_zero, f"+{without_zero}"])
        
        # Try extracting country code (first 1-3 digits) and national number
        if len(clean_phone) >= 8:
            for cc_len in [1, 2, 3]:
                if len(clean_phone) > cc_len:
                    country_code = clean_phone[:cc_len]
                    national_number = clean_phone[cc_len:]
                    
                    # Add variations with/without country code
                    variations.extend([
                        f"+{country_code}{national_number}",
                        f"{country_code}{national_number}",
                        national_number,
                    ])
                    
                    # Handle national format with leading zero
                    if not national_number.startswith('0'):
                        variations.append(f"0{national_number}")
        
        # Remove duplicates and empty values
        unique_variations = []
        seen = set()
        for var in variations:
            if var and var not in seen and len(var.replace('+', '')) >= 6:
                seen.add(var)
                unique_variations.append(var)
        
        logger.info(f"📱 Trying {len(unique_variations)} phone variations")
        
        # Try each variation
        for i, variation in enumerate(unique_variations):
            try:
                patient = self._make_request("GET", f"/patients/by-phone/{variation}")
                if patient:
                    logger.info(f"✅ Found patient: '{variation}' -> ID: {patient.get('id')}")
                    return patient
            except Exception as e:
                logger.debug(f"No match for '{variation}': {e}")
                continue
        
        # Fallback: If phone lookup failed, try getting user by phone, then patient by user_id
        logger.info(f"⚠️ Phone lookup failed, trying fallback: user lookup then patient by user_id")
        try:
            user = self.get_user_by_phone_number(phone_number)
            if user and user.get("id"):
                user_id = user.get("id")
                logger.info(f"✅ Found user: {user_id}, now fetching patient by user_id")
                patient = self._make_request("GET", f"/patients/user/{user_id}")
                if patient:
                    logger.info(f"✅ Found patient via fallback: user_id {user_id} -> patient ID: {patient.get('id')}")
                    return patient
        except Exception as e:
            logger.debug(f"Fallback lookup failed: {e}")
        
        logger.info(f"❌ No patient found for {phone_number}")
        return None
    def update_patient_language_preference_by_phone(self, phone_number: str, language_preference: str) -> Optional[Dict[str, Any]]:
        """
        Updates a patient's language preference using their phone number.
        This method finds the patient first, then updates their associated user's language preference.
        
        Args:
            phone_number: The patient's phone number
            language_preference: The new language preference (e.g., 'en', 'ar', 'fr')
        
        Returns:
            Dict with success/error information and updated data
        """
        logger.info(f"Attempting to update language preference to '{language_preference}' for patient with phone: {phone_number}")
        
        # First, find the patient by phone number
        patient = self.get_patient_by_phone_number(phone_number)
        if not patient:
            logger.error(f"Patient with phone number {phone_number} not found. Cannot update language preference.")
            return {
                "success": False,
                "error": "patient_not_found",
                "message": f"No patient found with phone number {phone_number}"
            }
        
        # Get the userId from the patient record (this matches the entity field name)
        user_id = patient.get("userId")
        patient_id = patient.get("id")
        patient_name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}"
        
        if not user_id:
            logger.error(f"Patient found but missing userId. Cannot update language preference.")
            return {
                "success": False,
                "error": "invalid_patient_data",
                "message": "Patient data is incomplete - missing user association"
            }
        
        logger.info(f"Found patient '{patient_name}' (ID: {patient_id}) with userId {user_id} for phone {phone_number}")
        
        # Check if user object is included (eager loaded) to get current language
        current_language = None
        if patient.get("user"):
            current_language = patient["user"].get("languagePreference", "Not set")
            
            # Check if language is already set to the requested value
            if current_language == language_preference:
                logger.info(f"Patient {patient_name} already has language preference set to '{language_preference}'. No update needed.")
                return {
                    "success": True,
                    "message": "Language preference already set to requested value",
                    "patient": patient,
                    "current_language": current_language
                }
        
        # Update the user's language preference
        update_payload = {
            "languagePreference": language_preference
        }
        
        try:
            updated_user = self._make_request("PATCH", f"/users/{user_id}", json_data=update_payload)
            if updated_user:
                logger.info(f"Successfully updated language preference to '{language_preference}' for patient '{patient_name}' (phone: {phone_number})")
                return {
                    "success": True,
                    "message": f"Language preference updated to '{language_preference}' for patient {patient_name}",
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "user_id": user_id,
                    "previous_language": current_language,
                    "new_language": language_preference,
                    "updated_user": updated_user
                }
            else:
                logger.error(f"Failed to update language preference for user {user_id} (patient '{patient_name}', phone: {phone_number})")
                return {
                    "success": False,
                    "error": "update_failed",
                    "message": "Server request failed to update user language preference"
                }
        except Exception as e:
            logger.error(f"Exception occurred while updating language preference for phone {phone_number}: {e}")
            return {
                "success": False,
                "error": "exception_occurred",
                "message": str(e)
            }

    def get_patient_with_user_details(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Gets patient information with full user details using their phone number.
        This is useful when you need both patient and user information together.
        
        Args:
            phone_number: The patient's phone number
            
        Returns:
            Dict with patient data including user details if found, None if not found
        """
        logger.info(f"Fetching patient with user details by phone number: {phone_number}")
        
        patient = self.get_patient_by_phone_number(phone_number)
        if not patient:
            return None
        
        # If user details are not included (not eager loaded), fetch them separately
        if not patient.get("user") and patient.get("userId"):
            user_details = self.get_user_by_id(patient["userId"])
            if user_details:
                patient["user"] = user_details
                logger.info(f"Added user details for patient {patient.get('id')}")
        
        return patient
    
    def get_user_by_phone_number(self, phone_number: str) -> Optional[Dict[str, Any]]:
        logger.warning(f"get_user_by_phone_number: Fetching all users to find by phone number '{phone_number}'. This is inefficient and should be optimized with a dedicated API endpoint in db-ops.")
        users = self._make_request("GET", "/users")
        if users and isinstance(users, list):
            logger.info(f"📱 DEBUG: Total users fetched: {len(users)}")
            
            # Create multiple phone number format variations to try
            phone_variations = []
            
            # Original phone number
            phone_variations.append(phone_number)
            
            # Clean version (remove all formatting and whitespace)
            clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '').replace('.', '').strip()
            phone_variations.append(clean_phone)
            
            # With + prefix
            if not phone_number.startswith('+'):
                phone_variations.append(f"+{clean_phone}")
            
            # With +971 for UAE numbers if not present
            if not clean_phone.startswith('971') and len(clean_phone) >= 9:
                phone_variations.append(f"+971{clean_phone}")
                phone_variations.append(f"971{clean_phone}")
            
            # Remove leading zeros and try again
            if clean_phone.startswith('0'):
                phone_without_zero = clean_phone[1:]
                phone_variations.append(phone_without_zero)
                phone_variations.append(f"+971{phone_without_zero}")
                phone_variations.append(f"971{phone_without_zero}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variations = []
            for phone in phone_variations:
                if phone not in seen:
                    seen.add(phone)
                    unique_variations.append(phone)
            
            logger.info(f"📱 DEBUG: Trying {len(unique_variations)} phone number variations: {unique_variations}")
            
            for i, user in enumerate(users):
                user_phone = user.get("phoneNumber")
                user_id = user.get("id", "N/A")
                user_email = user.get("email", "N/A")
                
                # Debug phone number details
                logger.info(f"📱 DEBUG: User {i+1}: id={user_id[:8] if user_id != 'N/A' else 'N/A'}..., email={user_email}")
                logger.info(f"📱 DEBUG: DB phone: '{user_phone}' (type: {type(user_phone)}, len: {len(user_phone) if user_phone else 0})")
                
                if user_phone:
                    # Clean the user's phone number from database
                    user_phone_clean = user_phone.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '').replace('.', '').strip()
                    
                    # Try matching against all phone variations
                    for phone_variant in unique_variations:
                        phone_variant_clean = phone_variant.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '').replace('.', '').strip()
                        
                        if user_phone_clean == phone_variant_clean:
                            logger.info(f"✅ Found user by phone number match: DB='{user_phone}' cleaned='{user_phone_clean}' matched variant='{phone_variant}' -> User ID: {user.get('id')}")
                            return user
                        
                        # Also try exact match for backward compatibility
                        if user_phone == phone_variant:
                            logger.info(f"✅ Found user by exact phone number match: '{user_phone}' == '{phone_variant}' -> User ID: {user.get('id')}")
                            return user
                    
                    logger.info(f"📱 DEBUG: No match for user {i+1} - DB phone cleaned: '{user_phone_clean}'")
                    
        logger.info(f"❌ User with phone number {phone_number} not found after checking all users and variations.")
        return None

    def get_user_preferences(self, patient_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Fetching preferences for patient ID: {patient_id}")
        return self._make_request("GET", f"/db/preferences/get_patient_preferences/{patient_id}")

    def create_patient_preferences(self, preference_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new patient preferences."""
        logger.info(f"DbOpsClient: Creating patient preferences with data: {preference_data}")
        # patient_id should be in preference_data as per CreatePatientPreferenceDto
        if not preference_data.get("patient_id"):
            logger.error("DbOpsClient.create_patient_preferences: patient_id missing in preference_data.")
            return None
        return self._make_request("POST", "/db/preferences/patient", json_data=preference_data)

    def update_patient_preferences_by_patient_id(self, patient_id: str, preference_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update patient preferences by patient ID."""
        logger.info(f"DbOpsClient: Updating preferences for patient ID: {patient_id} with data: {preference_data}")
        return self._make_request("PATCH", f"/db/preferences/update_patient_preferences/{patient_id}", json_data=preference_data)

    def get_clinic_info(self, clinic_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # This fetches clinic general info by clinic_id, or first clinic's info if no ID is provided.
        logger.info(f"Fetching clinic info. Clinic ID: {clinic_id if clinic_id else 'default'}")
        if clinic_id:
            return self._make_request("GET", f"/clinics/{clinic_id}")
        else:
            # If no clinic_id is provided, fetch the first clinic's info
            clinics = self._make_request("GET", "/clinics")
            if clinics and isinstance(clinics, list) and len(clinics) > 0:
                return clinics[0]
            else:
                logger.warning("No clinics found when trying to get default clinic info.")
                return None
    
    def get_all_clinics(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch all available clinic branches."""
        logger.info("Fetching all clinic branches")
        return self._make_request("GET", "/clinics")

    def get_patient_appointments(
        self, 
        patient_id: str, 
        appointment_date: Optional[str] = None,
        start_time: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get appointments for a patient, optionally filtered by date and time.
        
        Args:
            patient_id: Patient's UUID
            appointment_date: Optional date in YYYY-MM-DD format to filter appointments
            start_time: Optional time in HH:MM format to filter appointments
        
        Returns:
            List of appointment dictionaries or None if error
        """
        logger.info(f"Fetching appointments for patient ID: {patient_id}, Date: {appointment_date}, Time: {start_time}")
        
        # Pass query parameters to API for server-side filtering
        params = {}
        if appointment_date:
            params["appointment_date"] = appointment_date
        if start_time:
            # Backend expects HH:MM format, will convert to HH:MM:SS if needed
            params["start_time"] = start_time
        
        appointments = self._make_request("GET", f"/appointments/patient/{patient_id}", params=params if params else None)
        
        # Client-side filtering as fallback (in case API doesn't support filtering yet)
        if appointments and (appointment_date or start_time):
            filtered = appointments
            if appointment_date:
                # Handle different date formats (YYYY-MM-DD or with time)
                filtered = [
                    apt for apt in filtered 
                    if (apt.get("appointment_date") == appointment_date or 
                        apt.get("date") == appointment_date or
                        (apt.get("appointment_date") and appointment_date in str(apt.get("appointment_date"))) or
                        (apt.get("date") and appointment_date in str(apt.get("date"))))
                ]
            if start_time:
                # Compare HH:MM part of time
                time_short = start_time[:5] if len(start_time) >= 5 else start_time
                filtered = [
                    apt for apt in filtered 
                    if (apt.get("start_time") and time_short == str(apt.get("start_time"))[:5]) or
                       (apt.get("time") and time_short == str(apt.get("time"))[:5])
                ]
            logger.info(f"Client-side filtered to {len(filtered)} appointments matching criteria")
            return filtered
        
        return appointments

    def get_insurance_providers(self) -> Optional[List[Dict[str, Any]]]:
        """Get all insurance providers from clinic_insurance_providers table."""
        logger.info("Fetching insurance providers.")
        return self._make_request("GET", "/clinics/insurance/providers")

    def get_payment_methods(self) -> Optional[List[Dict[str, Any]]]:
        """Get all payment methods from clinic_payment_methods table."""
        logger.info("Fetching payment methods.")
        return self._make_request("GET", "/clinics/payment/methods")

    def get_visit_type_fees(self) -> Optional[List[Dict[str, Any]]]:
        """Get visit type fees - using visit-fees endpoint instead of visit-fees."""
        logger.info("Fetching visit type fees.")
        return self._make_request("GET", "/clinics/visit-fees")

    # src/patient_ai_service/infrastructure/db_ops_client.py

    async def add_communication_log(
        self,
        patient_id: str,
        message: str,
        *,
        user_id: str | None = None,
        doctor_id: str | None = None,
        message_type: str = "text",
        channel: str = "whatsapp",
        direction: str = "outbound",
        intent: str | None = None,
    ) -> dict:
        """
        Persist a communication row in db-ops so it appears in the dashboard.
        """
        import httpx
        
        # Ensure we have a valid token
        if not self.auth_token:
            self._get_auth_token()
        
        url = f"{self.base_url}/communication-logs"
        payload = {
            "patient_id": patient_id,
            "user_id": user_id,
            "doctor_id": doctor_id,
            "message": message,
            "message_type": message_type,
            "channel": channel,
            "direction": direction,
            "intent": intent,
        }
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 401:
                    # Token may be stale, try to refresh and retry
                    if self._refresh_access_token():
                        headers["Authorization"] = f"Bearer {self.auth_token}"
                        resp = await client.post(url, json=payload, headers=headers)
                
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"Error adding communication log: {e}")
            raise

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Fetching user by ID: {user_id}")
        return self._make_request("GET", f"/users/{user_id}")

    def register_user(self, email: str, full_name: str, phone_number: str, role_id: str, 
                      password: Optional[str] = None, # Made password optional
                      username: Optional[str] = None, 
                      language_preference: Optional[str] = "en") -> Optional[Dict[str, Any]]:
        """
        Registers a new user with the db-ops service.
        If password is None, db-ops must support passwordless registration, otherwise it will fail.
        """
        logger.info(f"Attempting to register new user: {email}, {full_name}")
        payload = {
            "email": email,
            "fullName": full_name,
            "phoneNumber": phone_number,
            "roleId": role_id,
            "languagePreference": language_preference
        }
        if username:
            payload["username"] = username
        else:
            # Attempt to generate username from email, then full_name, then a default
            if email and '@' in email:
                payload["username"] = email.split('@')[0]
            elif full_name:
                payload["username"] = full_name.replace(" ", "_").lower() + "_user"
            else:
                # Fallback if both email and full_name are insufficient or None
                # Consider using a portion of the phone_number or a random string if appropriate
                # For now, a generic username or raise an error if email/full_name are critical for username
                logger.warning("Email is None or invalid, and full_name is not provided. Cannot generate preferred username. A default may be used by the API or cause an error.")
                # If db-ops API requires username and cannot generate one, this call might still fail.
                # Depending on requirements, you might assign a random username or skip setting it.
                # For this example, we'll let it proceed, db-ops might assign one or error out if it's required and not generated.
                # payload["username"] = "default_user_" + phone_number[-4:] # Example if phone_number is always available
            
        if password:
            payload["password"] = password
        else:
            # If password is None, we omit it. 
            # db-ops' RegisterDto currently requires password. This call will likely fail
            # if db-ops is not updated to make password optional for certain registration types.
            logger.warning(f"Attempting to register user {email} without a password. DB-Ops must support this.")

        current_token = self.auth_token 
        self.auth_token = None 

        try:
            response = self._make_request("POST", "/auth/register", json_data=payload, attempts=1) 
        finally:
            self.auth_token = current_token 

        if response:
            logger.info(f"User {email} registration attempt processed. Response: {response}")
        else:
            logger.error(f"User registration attempt for {email} failed or received no response.")
        
        return response

    def get_patient_by_user_id(self, user_id: str) -> Optional[Dict]:
        logger.info(f"DbOpsClient: Getting patient details by their associated user ID: {user_id}")
        return self._make_request("GET", f"/patients/user/{user_id}") # Assuming this is the correct endpoint

    def get_all_users(self) -> Optional[List[Dict[str, Any]]]:
        logger.info("Fetching all users.")
        return self._make_request("GET", "/users")

    def create_patient(self, user_id: str, first_name: str, last_name: str, date_of_birth: Optional[str] = None, # Made optional to match DTO more closely
                       gender: Optional[str] = None, # Made optional
                       emergency_contact_name: Optional[str] = None, 
                       emergency_contact_phone: Optional[str] = None, insurance_provider: Optional[str] = None, 
                       insurance_policy_number: Optional[str] = None, 
                       medical_history: Optional[Dict[str, Any]] = None, allergies: Optional[List[str]] = None, 
                       medications: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Creates a new patient record associated with an existing user.
        Matches the CreatePatientDto in db-ops.
        date_of_birth should be in 'YYYY-MM-DD' format if provided.
        """
        logger.info(f"Attempting to create patient record for user_id: {user_id}")
        payload = {
            "userId": user_id,
            "first_name": first_name,  # DTO uses snake_case
            "last_name": last_name,    # DTO uses snake_case
            # Optional fields, ensure they are not None if not allowed by DTO, or handle if API expects them to be absent
            "date_of_birth": date_of_birth,  # DTO uses snake_case, format: YYYY-MM-DD
            "gender": gender,
            "emergency_contact": emergency_contact_name,  # DTO uses emergency_contact (not emergencyContactName)
            "emergency_phone": emergency_contact_phone,    # DTO uses emergency_phone (not emergencyContactPhone)
            "insurance_provider": insurance_provider,      # DTO uses snake_case
            "insurance_policy_number": insurance_policy_number,  # DTO uses snake_case
            "medical_history": medical_history if medical_history is not None else {},
            "allergies": allergies if allergies is not None else [],
            "medications": medications if medications is not None else []
        }
        # Remove keys with None values, as API might prefer them absent vs. null
        # unless the DTO explicitly allows null for optional fields.
        # Based on CreatePatientDto, most are optional strings/arrays/JSON, so sending empty/default is fine.
        # For truly optional fields that should be absent if not provided, a cleanup loop would be here.
        
        logger.info(f"DbOpsClient: Sending payload to create patient: {payload}") # ADDED: Log payload
        response_data = self._make_request("POST", "/patients", json_data=payload)
        logger.info(f"DbOpsClient: Response from create patient: {response_data}") # ADDED: Log response
        return response_data

    # --- New methods based on API Documentation ---

    # Users and Roles
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Gets the currently authenticated user."""
        logger.info("Fetching current authenticated user.")
        return self._make_request("GET", "/auth/me")

    def get_user_roles(self) -> Optional[List[Dict[str, Any]]]:
        """Gets available user roles."""
        logger.info("Fetching user roles.")
        return self._make_request("GET", "/users/roles")

    def get_user_permissions(self) -> Optional[List[Dict[str, Any]]]:
        """Gets available user permissions."""
        logger.info("Fetching user permissions.")
        return self._make_request("GET", "/users/permissions")

    # Clinics and Settings
    def create_clinic(self, name: str, address: str, city: str, country: str, phone: str, email: str, website: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Creates a new clinic."""
        logger.info(f"Creating new clinic: {name}")
        payload = {
            "name": name,
            "address": address,
            "city": city,
            "country": country,
            "phone": phone,
            "email": email,
            "website": website
        }
        return self._make_request("POST", "/clinics", json_data=payload)

    def get_clinic_settings(self, clinic_id: str) -> Optional[List[Dict[str, Any]]]:
        """Gets all settings for a specific clinic."""
        logger.info(f"Fetching settings for clinic ID: {clinic_id}")
        return self._make_request("GET", f"/clinics/{clinic_id}/settings")

    def get_clinic_setting_by_key(self, clinic_id: str, key: str) -> Optional[Dict[str, Any]]:
        """Gets a specific setting for a clinic by key."""
        logger.info(f"Fetching setting '{key}' for clinic ID: {clinic_id}")
        return self._make_request("GET", f"/clinics/{clinic_id}/settings/{key}")

    def create_or_update_clinic_setting(self, clinic_id: str, setting_key: str, setting_value: Any) -> Optional[Dict[str, Any]]:
        """Creates or updates a clinic setting."""
        logger.info(f"Creating/updating setting '{setting_key}' for clinic ID: {clinic_id}")
        payload = {
            "clinic_id": clinic_id,
            "setting_key": setting_key,
            "setting_value": setting_value
        }
        return self._make_request("POST", "/clinics/settings", json_data=payload)

    def get_clinic_appointments(self, clinic_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Gets appointments for a specific clinic, optionally filtered by date."""
        logger.info(f"Fetching appointments for clinic ID: {clinic_id}")
        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date
        return self._make_request("GET", f"/clinics/{clinic_id}/appointments", params=params)

    def get_clinic_appointment_policy(self, clinic_id: str) -> Optional[Dict[str, Any]]:
        """Get clinic appointment policy using settings endpoint since appointment-policy doesn't exist."""
        logger.info(f"Fetching appointment policy for clinic ID: {clinic_id}")
        # Use clinic settings instead since appointment-policy endpoint doesn't exist
        try:
            settings = self._make_request("GET", f"/clinics/{clinic_id}/appointment-policy")
            if settings:
                # Look for appointment policy in settings
                return settings
            return None
        except Exception as e:
            logger.warning(f"Could not fetch appointment policy for clinic {clinic_id}: {e}")
            return None

    def create_or_update_clinic_appointment_policy(self, clinic_id: str, policy_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Creates or updates the appointment policy for a specific clinic."""
        logger.info(f"Creating/updating appointment policy for clinic ID: {clinic_id}")
        return self._make_request("POST", f"/clinics/{clinic_id}/appointment-policy", json_data=policy_data)

    # Doctors and Specialties
    def add_doctor_availability(self, doctor_id: str, availability_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a new availability time slot for a doctor."""
        logger.info(f"Adding availability for doctor ID: {doctor_id}")
        # The DTO in API doc has doctor_id in body, but endpoint also takes it in path.
        # Assuming data in availability_data is structured as per POST /doctors/{id}/availability
        return self._make_request("POST", f"/doctors/{doctor_id}/availability", json_data=availability_data)

    def update_doctor_availability(self, doctor_id: str, availability_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates an existing availability time slot for a doctor."""
        logger.info(f"Updating availability ID {availability_id} for doctor ID: {doctor_id}")
        return self._make_request("PATCH", f"/doctors/{doctor_id}/availability/{availability_id}", json_data=update_data)

    def delete_doctor_availability(self, doctor_id: str, availability_id: str) -> Optional[Any]: # 204 No Content
        """Deletes an availability time slot for a doctor."""
        logger.info(f"Deleting availability ID {availability_id} for doctor ID: {doctor_id}")
        return self._make_request("DELETE", f"/doctors/{doctor_id}/availability/{availability_id}")

    # Appointments
    def get_all_appointments(self) -> Optional[List[Dict[str, Any]]]:
        """Gets a list of all appointments."""
        logger.info("Fetching all appointments.")
        return self._make_request("GET", "/appointments")
        
    def mark_appointment_no_show(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Marks an appointment as a no-show."""
        logger.info(f"Marking appointment ID {appointment_id} as no-show.")
        return self._make_request("PATCH", f"/appointments/{appointment_id}/no-show")

    def get_appointment_types(self) -> Optional[List[Dict[str, Any]]]:
        """Gets all appointment types."""
        logger.info("Fetching all appointment types.")
        return self._make_request("GET", "/appointment-types")

    # Patients
    def get_all_patients(self) -> Optional[List[Dict[str, Any]]]:
        """Gets a list of all patients."""
        logger.info("Fetching all patients.")
        return self._make_request("GET", "/patients")

    def get_appointments_for_patient(self, patient_id: str) -> Optional[List[Dict[str, Any]]]:
        """Gets appointments for a specific patient using patient ID in path.
        This aligns with GET /patients/{id}/appointments from API documentation.
        Note: an existing method get_patient_appointments uses GET /appointments/patient/{patient_id}
        """
        logger.info(f"Fetching appointments for patient ID {patient_id} (using /patients/{patient_id}/appointments).")
        return self._make_request("GET", f"/patients/{patient_id}/appointments")

    # Waitlist
    def get_all_waitlist_entries(self) -> Optional[List[Dict[str, Any]]]:
        """Gets all waitlist entries."""
        logger.info("Fetching all waitlist entries.")
        return self._make_request("GET", "/db/waitlist")

    def get_waitlist_entries_by_status(self, status: str) -> Optional[List[Dict[str, Any]]]:
        """Gets waitlist entries filtered by status."""
        logger.info(f"Fetching waitlist entries with status: {status}")
        return self._make_request("GET", f"/db/waitlist/status/{status}")

    def add_to_waitlist(self, waitlist_entry_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a patient to the waitlist."""
        logger.info(f"Adding entry to waitlist for patient ID: {waitlist_entry_data.get('patient_id')}")
        return self._make_request("POST", "/db/waitlist/add", json_data=waitlist_entry_data)

    # Preferences (User-specific as per API doc)
    def get_user_preferences_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Gets preferences for a specific user by user ID."""
        logger.info(f"Fetching preferences for user ID: {user_id} (using /preferences/user/{user_id})")
        return self._make_request("GET", f"/preferences/user/{user_id}")

    def update_user_preferences_by_user_id(self, user_id: str, preferences_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates preferences for a specific user by user ID."""
        logger.info(f"Updating preferences for user ID: {user_id} (using /preferences/user/{user_id})")
        return self._make_request("PATCH", f"/preferences/user/{user_id}", json_data=preferences_data)

    # Reminders
    def get_all_reminders(self) -> Optional[List[Dict[str, Any]]]:
        """Gets all reminders."""
        logger.info("Fetching all reminders.")
        return self._make_request("GET", "/db/reminders")

    def get_due_reminders(self) -> Optional[List[Dict[str, Any]]]:
        """Gets reminders that are due to be sent."""
        logger.info("Fetching due reminders.")
        return self._make_request("GET", "/db/reminders/get_due")

    def create_reminder(self, reminder_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Creates a new reminder."""
        logger.info(f"Creating reminder for patient ID: {reminder_data.get('patient_id')}")
        return self._make_request("POST", "/db/reminders/create", json_data=reminder_data)

    # Insurance and Payment
    def get_clinic_accepted_insurances(self, clinic_id: str) -> Optional[Dict[str, Any]]: # Response has a specific structure
        """Gets all insurance providers accepted by a specific clinic."""
        logger.info(f"Fetching accepted insurances for clinic ID: {clinic_id}")
        return self._make_request("GET", f"/clinics/{clinic_id}/accepted-insurances")

    def get_clinic_insurance_providers(self, clinic_id: str) -> Optional[List[Dict[str, Any]]]:
        """Gets insurance providers for a specific clinic with network coverage details."""
        logger.info(f"Fetching insurance providers for clinic ID: {clinic_id}")
        return self._make_request("GET", f"/clinics/{clinic_id}/insurance/providers")

    def get_all_clinics_insurance_providers(self) -> Optional[List[Dict[str, Any]]]:
        """Gets insurance providers for all clinics with branch-specific details."""
        logger.info("Fetching insurance providers for all clinics")
        return self._make_request("GET", "/clinics/insurance/providers/all")

    def get_insurance_pre_authorization_process(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Gets pre-authorization process details for a specific insurance provider."""
        logger.info(f"Fetching pre-authorization process for provider: {provider_name}")
        return self._make_request("GET", f"/clinics/insurance/pre-authorization/{provider_name}")

    def get_procedure_insurance_coverage(self, procedure_id: str, insurance_id: str) -> Optional[Dict[str, Any]]:
        """Gets insurance coverage information for a specific dental procedure."""
        logger.info(f"Fetching insurance coverage for procedure ID {procedure_id} and insurance ID {insurance_id}")
        params = {"procedureId": procedure_id, "insuranceId": insurance_id}
        return self._make_request("GET", "/clinics/procedures/insurance-coverage", params=params)

    def get_all_dental_procedures(self) -> Optional[List[Dict[str, Any]]]:
        """Gets a list of all available dental procedures with pricing information."""
        logger.info("Fetching all dental procedures.")
        return self._make_request("GET", "/procedures")

    def get_procedures_by_specialty(self, specialty_id: str) -> Optional[List[Dict[str, Any]]]:
        """Gets dental procedures for a specific specialty."""
        logger.info(f"Fetching procedures for specialty ID: {specialty_id}")
        return self._make_request("GET", f"/procedures/specialty/{specialty_id}")

    def get_discounts_and_promotions(self) -> Optional[List[Dict[str, Any]]]:
        """Gets all active discounts and promotions."""
        logger.info("Fetching discounts and promotions.")
        # API doc path is /clinics/discounts, implies it might be clinic-specific or general.
        # Assuming general for now as no clinic_id in path.
        return self._make_request("GET", "/clinics/discounts")

    def get_payment_policies(self) -> Optional[List[Dict[str, Any]]]:
        """Gets all payment policies."""
        logger.info("Fetching payment policies.")
        # API doc path is /clinics/payment-policies.
        return self._make_request("GET", "/clinics/payment-policies")

    def get_clinic_procedures(self, clinic_id: str) -> Optional[Dict[str, Any]]: # Response has specific structure
        """Gets all dental procedures available at a specific clinic."""
        logger.info(f"Fetching procedures for clinic ID: {clinic_id}")
        return self._make_request("GET", f"/clinics/{clinic_id}/procedures")

    # Emergencies
    def report_emergency(self, emergency_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reports a new emergency."""
        logger.info(f"Reporting new emergency for clinic ID: {emergency_data.get('clinicId')}")
        return self._make_request("POST", "/emergencies", json_data=emergency_data)

    def get_all_emergencies(self, clinic_id: Optional[str] = None, status: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Retrieves a list of all reported emergencies, optionally filtered."""
        logger.info("Fetching all emergencies.")
        params = {}
        if clinic_id:
            params["clinicId"] = clinic_id
        if status:
            params["status"] = status
        return self._make_request("GET", "/emergencies", params=params if params else None)

    def get_emergency_by_id(self, emergency_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves details of a specific emergency."""
        logger.info(f"Fetching emergency by ID: {emergency_id}")
        return self._make_request("GET", f"/emergencies/{emergency_id}")

    def update_emergency_status(self, emergency_id: str, status: str, notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Updates the status of an existing emergency."""
        logger.info(f"Updating status for emergency ID: {emergency_id} to {status}")
        payload = {"status": status}
        if notes is not None: # Ensure notes is only added if provided
            payload["notes"] = notes
        return self._make_request("PUT", f"/emergencies/{emergency_id}/status", json_data=payload)

    def update_emergency_priority(self, emergency_id: str, priority: str) -> Optional[Dict[str, Any]]:
        """Updates the priority of an existing emergency."""
        logger.info(f"Updating priority for emergency ID: {emergency_id} to {priority}")
        payload = {"priority": priority}
        return self._make_request("PUT", f"/emergencies/{emergency_id}/priority", json_data=payload)

    # Inquiries
    def create_inquiry(self, inquiry_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Creates a new medical inquiry."""
        logger.info(f"Creating new inquiry for patient ID: {inquiry_data.get('patientId')}")
        logger.info(f"Inquiry data: {inquiry_data}")
        return self._make_request("POST", "/inquiries", json_data=inquiry_data)

    def update_user_language_preference_by_phone(self, phone_number: str, language_preference: str) -> Optional[Dict[str, Any]]:
        """
        Updates a user's language preference using their phone number.
        
        Args:
            phone_number: The user's phone number
            language_preference: The new language preference (e.g., 'en', 'ar', 'fr')
        
        Returns:
            Dict with updated user data if successful, None if failed
        """
        logger.info(f"Attempting to update language preference to '{language_preference}' for phone number: {phone_number}")
        
        # First, find the user by phone number
        user = self.get_user_by_phone_number(phone_number)
        if not user:
            logger.error(f"User with phone number {phone_number} not found. Cannot update language preference.")
            return None
        
        user_id = user.get("id")
        if not user_id:
            logger.error(f"User found but missing ID. Cannot update language preference.")
            return None
        
        logger.info(f"Found user ID {user_id} for phone number {phone_number}. Updating language preference...")
        
        # Prepare the update payload
        update_payload = {
            "languagePreference": language_preference
        }
        
        # Update the user using the existing PATCH /users/{id} endpoint
        try:
            updated_user = self._make_request("PATCH", f"/users/{user_id}", json_data=update_payload)
            if updated_user:
                logger.info(f"Successfully updated language preference to '{language_preference}' for user {user_id} (phone: {phone_number})")
                return updated_user
            else:
                logger.error(f"Failed to update language preference for user {user_id} (phone: {phone_number})")
                return None
        except Exception as e:
            logger.error(f"Exception occurred while updating language preference for phone {phone_number}: {e}")
            return None
    def get_all_inquiries(self, inquiry_type: Optional[str] = None, priority: Optional[str] = None, unanswered: Optional[bool] = None) -> Optional[List[Dict[str, Any]]]:
        """Gets all inquiries with optional filters."""
        logger.info("Fetching all inquiries.")
        params = {}
        if inquiry_type:
            params["type"] = inquiry_type
        if priority:
            params["priority"] = priority
        if unanswered is not None:
            params["unanswered"] = "true" if unanswered else "false"
        return self._make_request("GET", "/inquiries", params=params if params else None)

    def get_inquiry_by_id(self, inquiry_id: str) -> Optional[Dict[str, Any]]:
        """Gets a specific inquiry by ID."""
        logger.info(f"Fetching inquiry by ID: {inquiry_id}")
        return self._make_request("GET", f"/inquiries/{inquiry_id}")

    def get_inquiries_by_patient_id(self, patient_id: str) -> Optional[List[Dict[str, Any]]]:
        """Gets all inquiries for a specific patient."""
        logger.info(f"Fetching inquiries for patient ID: {patient_id}")
        return self._make_request("GET", f"/inquiries/patient/{patient_id}")

    def update_inquiry(self, inquiry_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates an existing inquiry."""
        logger.info(f"Updating inquiry ID: {inquiry_id}")
        return self._make_request("PATCH", f"/inquiries/{inquiry_id}", json_data=update_data)

    def mark_inquiry_as_answered(self, inquiry_id: str, answer_text: str, answered_by_user_id: str) -> Optional[Dict[str, Any]]:
        """Marks an inquiry as answered."""
        logger.info(f"Marking inquiry ID {inquiry_id} as answered")
        payload = {
            "answerText": answer_text,
            "answeredByUserId": answered_by_user_id
        }
        return self._make_request("PATCH", f"/inquiries/{inquiry_id}/answer", json_data=payload)

    def delete_inquiry(self, inquiry_id: str) -> Optional[Any]:
        """Deletes an inquiry."""
        logger.info(f"Deleting inquiry ID: {inquiry_id}")
        return self._make_request("DELETE", f"/inquiries/{inquiry_id}")

    def get_inquiry_stats(self) -> Optional[Dict[str, Any]]:
        """Gets inquiry statistics."""
        logger.info("Fetching inquiry statistics.")
        return self._make_request("GET", "/inquiries/stats")

    def get_doctor_by_user_id(self, user_id: str) -> Optional[Dict]:
        logger.info(f"DbOpsClient: Getting doctor details by their associated user ID: {user_id}")
        return self._make_request("GET", f"/doctors/user/{user_id}") # Using the correct endpoint from doctor controller

    # --- NEW PROCEDURE AND PROCEDURE GUIDELINES METHODS ---

    def get_procedure_by_name(self, procedure_name: str) -> Optional[List[Dict[str, Any]]]:
        """Gets procedures by name (supports partial matching)."""
        logger.info(f"Fetching procedures by name: {procedure_name}")
        return self._make_request("GET", f"/procedures/name/{procedure_name}")

    def get_procedure_by_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Gets a specific procedure by ID."""
        logger.info(f"Fetching procedure by ID: {procedure_id}")
        return self._make_request("GET", f"/procedures/{procedure_id}")

    # Procedure Guidelines Methods
    def get_procedure_guidelines_by_procedure_name(self, procedure_name: str) -> Optional[Dict[str, Any]]:
        """Gets procedure guidelines by procedure name."""
        logger.info(f"Fetching procedure guidelines by name: {procedure_name}")
        return self._make_request("GET", f"/procedure-guidelines/procedure/{procedure_name}")

    def get_procedure_guidelines_by_procedure_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Gets procedure guidelines by procedure ID."""
        logger.info(f"Fetching procedure guidelines by procedure ID: {procedure_id}")
        return self._make_request("GET", f"/procedure-guidelines/procedure-id/{procedure_id}")

    def get_pre_visit_guidelines_by_procedure_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Gets pre-visit guidelines by procedure ID."""
        logger.info(f"Fetching pre-visit guidelines for procedure ID: {procedure_id}")
        return self._make_request("GET", f"/procedure-guidelines/pre-visit/{procedure_id}")

    def get_post_visit_guidelines_by_procedure_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Gets post-visit guidelines by procedure ID."""
        logger.info(f"Fetching post-visit guidelines for procedure ID: {procedure_id}")
        return self._make_request("GET", f"/procedure-guidelines/post-visit/{procedure_id}")

    def get_pre_visit_questions_by_procedure_id(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Gets pre-visit questions by procedure ID."""
        logger.info(f"Fetching pre-visit questions for procedure ID: {procedure_id}")
        return self._make_request("GET", f"/procedure-guidelines/questions/{procedure_id}")

    # Appointment-based Guidelines Methods
    def get_appointment_pre_visit_questions(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Gets pre-visit questions for an appointment."""
        logger.info(f"Fetching pre-visit questions for appointment ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/questions/{appointment_id}")

    def get_appointment_pre_visit_guidelines(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Gets pre-visit guidelines for an appointment."""
        logger.info(f"Fetching pre-visit guidelines for appointment ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/pre-visit/{appointment_id}")

    def get_appointment_post_visit_guidelines(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Gets post-visit guidelines for an appointment."""
        logger.info(f"Fetching post-visit guidelines for appointment ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/post-visit/{appointment_id}")

    def get_appointment_all_procedure_info(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Gets all procedure information for an appointment."""
        logger.info(f"Fetching all procedure info for appointment ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/all/{appointment_id}")

    def get_appointment_procedure_summary(self, appointment_id: str) -> Optional[Dict[str, Any]]:
        """Gets appointment procedure summary."""
        logger.info(f"Fetching appointment procedure summary for appointment ID: {appointment_id}")
        return self._make_request("GET", f"/appointments/{appointment_id}/summary")

    # --- END NEW METHODS ---

# Example usage section
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("DB Ops Client - Example Usage (requires db-ops service running and configured .env)")

    # Set dummy env vars for testing if not present (REMOVE FOR PRODUCTION if hardcoding like this)
    # It's better to set these in your shell environment or a .env file loaded by your application's entry point.
    os.environ.setdefault("DB_OPS_USER_EMAIL", "admin@healthcareclinic.com") 
    os.environ.setdefault("DB_OPS_USER_PASSWORD", "yourStrongPassword")      
    # os.environ.setdefault("DB_OPS_URL", "http://localhost:3000/api") # Already handled with default in constructor

    # Instantiate the client
    # You can pass credentials directly: client = DbOpsClient(user_email="test@example.com", user_password="password")
    client = DbOpsClient()

    if not client.auth_token:
        logger.error("Exiting example usage due to initial authentication failure.")
        exit()

    logger.info("\n--- Testing get_doctors ---")
    doctors = client.get_doctors()
    sample_doctor_id = None
    if doctors:
        logger.info(f"Found {len(doctors)} doctors.")
        if len(doctors) > 0:
            sample_doctor_id = doctors[0].get("id")
            logger.info(f"Sample doctor: {doctors[0].get('first_name')} {doctors[0].get('last_name')}, ID: {sample_doctor_id}")
            
            logger.info("\n--- Testing get_doctor_by_id ---")
            if sample_doctor_id:
                doctor_detail = client.get_doctor_by_id(sample_doctor_id)
                if doctor_detail:
                    logger.info(f"Details for doctor ID {sample_doctor_id}: {doctor_detail.get('first_name')} {doctor_detail.get('last_name')}, Bio: {doctor_detail.get('bio')}")

            logger.info("\n--- Testing get_doctor_availability ---")
            if sample_doctor_id:
                availability = client.get_doctor_availability(doctor_id=sample_doctor_id) #, date="2024-07-30"
                if availability:
                    logger.info(f"Found availability for doctor {sample_doctor_id}: {availability}")
                else:
                    logger.warning(f"No availability found or error for doctor {sample_doctor_id}.")
        else:
            logger.info("No doctors found to test further doctor-specific endpoints.")
    else:
        logger.warning("Could not fetch doctors.")

    logger.info("\n--- Testing get_specialties ---")
    specialties = client.get_specialties()
    if specialties:
        logger.info(f"Found {len(specialties)} specialties. First one: {specialties[0] if specialties else 'N/A'}")
    else:
        logger.warning("Could not fetch specialties.")

    logger.info("\n--- Testing get_available_doctors ---")
    available_docs = client.get_available_doctors(date="2024-12-01", start_time="09:00", end_time="17:00")
    if available_docs:
        logger.info(f"Found {len(available_docs)} available doctors for the specified slot.")
    else:
        logger.warning("No available doctors found for the slot or error occurred.")

    logger.info("\n--- Testing get_clinic_info ---")
    clinic_info = client.get_clinic_info() 
    if clinic_info:
        logger.info(f"Clinic Info: {clinic_info.get('name')}")
    else:
        logger.warning("Could not fetch clinic info.")

    if doctors and len(doctors) > 0 and clinic_info and sample_doctor_id:
        test_patient_id = "patient-uuid-placeholder" 
        test_doctor_id = sample_doctor_id
        test_clinic_id = clinic_info.get("id")

        if test_patient_id != "patient-uuid-placeholder" and test_doctor_id and test_clinic_id:
            logger.info("\n--- Testing create_appointment ---")
            # Replace test_patient_id with a REAL one from your DB to make this pass
            created_app = client.create_appointment(
                clinic_id=test_clinic_id,
                patient_id=test_patient_id, 
                doctor_id=test_doctor_id, 
                appointment_date="2024-12-25", 
                start_time="14:00", 
                end_time="14:30", 
                status="scheduled", 
                notes="Test appt from db_ops_client"
            )
            sample_appointment_id = None
            if created_app:
                logger.info(f"Appointment created: {created_app.get('id')}")
                sample_appointment_id = created_app.get('id')

                if sample_appointment_id:
                    logger.info("\n--- Testing get_appointment_by_id ---")
                    appt_details = client.get_appointment_by_id(sample_appointment_id)
                    if appt_details:
                        logger.info(f"Fetched appointment details: {appt_details}")
                    
                    logger.info("\n--- Testing reschedule_appointment ---")
                    rescheduled_app = client.reschedule_appointment(sample_appointment_id, "2024-12-26", "15:00", "15:30", "Patient request")
                    if rescheduled_app:
                         logger.info(f"Appointment rescheduled: {rescheduled_app}")

                    logger.info("\n--- Testing cancel_appointment ---")
                    cancelled_success = client.cancel_appointment(sample_appointment_id, "Patient no longer available")
                    if cancelled_success:
                        logger.info(f"Appointment {sample_appointment_id} cancellation request processed.")
                    else:
                        logger.warning(f"Appointment {sample_appointment_id} cancellation failed.")
            else:
                logger.warning("Appointment creation failed. (Is patient-uuid-placeholder a valid ID in your DB?)")
        else:
            logger.warning("Skipping appointment creation tests due to missing test IDs (patient, doctor, or clinic).")
    else:
        logger.warning("Skipping appointment tests because no doctors/clinic info or sample_doctor_id could be fetched/determined.")

    logger.info("\n--- Testing get_insurance_providers ---")
    insurance_providers = client.get_insurance_providers()
    if insurance_providers:
        logger.info(f"Found {len(insurance_providers)} insurance providers.")
    else:
        logger.warning("Could not fetch insurance providers.")

    logger.info("\n--- Testing get_payment_methods ---")
    payment_methods = client.get_payment_methods()
    if payment_methods:
        logger.info(f"Found {len(payment_methods)} payment methods.")
    else:
        logger.warning("Could not fetch payment methods.")

    logger.info("\n--- Testing get_clinic_fees ---")
    clinic_fees = client.get_visit_type_fees()
    if clinic_fees:
        logger.info(f"Found {len(clinic_fees)} clinic fee entries.")
    else:
        logger.warning("Could not fetch clinic fees.")
        
    logger.info("\n--- db_ops_client example usage finished ---") 