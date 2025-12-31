"""
Notification Service Client for Patient AI Service

This client handles all communication with the notification service,
including appointment lifecycle notifications, reminders, and patient responses.
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationServiceClient:
    """Client for communicating with the Notification Service from Patient AI Service."""
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the notification service client.
        
        Args:
            base_url: Base URL of the notification service
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv(
            "NOTIFICATION_SERVICE_URL", 
            "http://notification-service:5000"
        )
        self.api_key = api_key or os.getenv(
            "NOTIFICATION_SERVICE_API_KEY",
            "ISMILE_CAREBOT_TIMEAI_3r7fANOk35Qj8bvZqMwmCDqSdQtB7oYJkr"
        )
        self.timeout = 30.0
        
        logger.info(f"NotificationServiceClient initialized with base_url: {self.base_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers
    
    async def send_notification(
        self,
        recipient_id: str,
        recipient_type: str,
        message: str,
        notification_type: str,
        appointment_id: Optional[str] = None,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send an immediate notification to a recipient.
        
        Args:
            recipient_id: ID of the recipient (patient_id or doctor_id)
            recipient_type: Type of recipient ("patient" or "doctor")
            message: Notification message content
            notification_type: Type of notification (e.g., "appointment_reminder", "appointment_confirmation")
            appointment_id: Optional appointment ID for context
            patient_id: Optional patient ID for context
            
        Returns:
            Response from notification service
        """
        try:
            payload = {
                "recipient_id": recipient_id,
                "recipient_type": recipient_type,
                "message": message,
                "notification_type": notification_type
            }
            
            if appointment_id:
                payload["appointment_id"] = appointment_id
            if patient_id:
                payload["patient_id"] = patient_id
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/notifications/send",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Notification sent successfully: {notification_type} to {recipient_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error sending notification: {e.response.status_code} - {e.response.text}")
            return {"status": "error", "message": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error sending notification: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def schedule_appointment_lifecycle_notifications(
        self,
        patient_id: str,
        appointment_id: str,
        procedure_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule automated lifecycle notifications for an appointment.
        
        This will schedule pre-visit questions, pre-visit guidelines, and post-visit guidelines.
        
        Args:
            patient_id: Patient ID
            appointment_id: Appointment ID
            procedure_type: Optional procedure type
            
        Returns:
            Response from notification service
        """
        try:
            payload = {
                "patient_id": patient_id,
                "appointment_id": appointment_id
            }
            
            if procedure_type:
                payload["procedure_type"] = procedure_type
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/appointment-lifecycle/schedule",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Lifecycle notifications scheduled for appointment {appointment_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error scheduling lifecycle notifications: {e.response.status_code} - {e.response.text}")
            return {"status": "error", "message": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error scheduling lifecycle notifications: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def create_appointment_reminder(
        self,
        appointment_id: str,
        message_template: Optional[str] = None,
        schedule_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an appointment reminder.
        
        Args:
            appointment_id: Appointment ID
            message_template: Optional custom message template
            schedule_time: Optional ISO format datetime string for when to send
            
        Returns:
            Response from notification service
        """
        try:
            payload = {
                "appointment_id": appointment_id
            }
            
            if message_template:
                payload["message_template"] = message_template
            if schedule_time:
                payload["schedule_time"] = schedule_time
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/notifications/appointment-reminder",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Appointment reminder created for {appointment_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error creating reminder: {e.response.status_code} - {e.response.text}")
            return {"status": "error", "message": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error creating reminder: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def process_patient_response(
        self,
        patient_id: str,
        message: str,
        phone_number: str
    ) -> Dict[str, Any]:
        """
        Process a patient response to a notification.
        
        This is used when patients respond to waitlist offers, appointment reminders, etc.
        
        Args:
            patient_id: Patient ID
            message: Patient's response message
            phone_number: Patient's phone number
            
        Returns:
            Response with processing result and context
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/patient/response",
                    params={
                        "patient_id": patient_id,
                        "message": message,
                        "phone_number": phone_number
                    },
                    headers=self._get_headers()
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"✅ Patient response processed for {patient_id}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error processing patient response: {e.response.status_code} - {e.response.text}")
            return {"status": "error", "message": f"HTTP {e.response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error processing patient response: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def send_appointment_confirmation(
        self,
        patient_id: str,
        appointment_id: str,
        appointment_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send an appointment confirmation notification.
        
        Args:
            patient_id: Patient ID
            appointment_id: Appointment ID
            appointment_details: Optional appointment details for message personalization
            
        Returns:
            Response from notification service
        """
        # Build confirmation message
        if appointment_details:
            doctor_name = appointment_details.get("doctor_name", "your doctor")
            date = appointment_details.get("date", "your appointment date")
            time = appointment_details.get("time", "your appointment time")
            message = (
                f"✅ Your appointment has been confirmed!\n\n"
                f"👨‍⚕️ Doctor: {doctor_name}\n"
                f"📅 Date: {date}\n"
                f"⏰ Time: {time}\n\n"
                f"Please arrive 15 minutes early."
            )
        else:
            message = "✅ Your appointment has been confirmed! Please arrive 15 minutes early."
        
        return await self.send_notification(
            recipient_id=patient_id,
            recipient_type="patient",
            message=message,
            notification_type="appointment_confirmation",
            appointment_id=appointment_id,
            patient_id=patient_id
        )
    
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of the notification service."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            return {"status": "error", "message": str(e)}



