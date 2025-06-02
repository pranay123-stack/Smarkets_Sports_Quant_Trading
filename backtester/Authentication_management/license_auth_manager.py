import requests
import smtplib
from email.mime.text import MIMEText
import jwt
import time
import hashlib
import redis
import threading
import random
from typing import Dict, Any, Optional
from auth_manager_interface import AuthManagerInterface
from smarkets_Sports_Quant_Trading.backtester.Logger.logger import Logger

class LicenseAuthManager(AuthManagerInterface):
    """Authentication manager for licensed backtester with multi-user support."""
    
    def __init__(self, license_server_url: str, secret_key: str, smtp_config: Dict[str, Any], 
                 redis_config: Dict[str, Any]):
        """Initialize the authentication manager.
        
        Args:
            license_server_url (str): URL of the license server.
            secret_key (str): Secret key for JWT encoding/decoding.
            smtp_config (Dict[str, Any]): SMTP configuration for email OTP.
            redis_config (Dict[str, Any]): Redis configuration for session management.
        """
        super().__init__()
        self.license_server_url = license_server_url
        self.secret_key = secret_key
        self.smtp_config = smtp_config
        self.logger = Logger(log_dir="Logs")
        self.otp_cache = {}  # Temporary OTP storage
        self.redis = redis.Redis(**redis_config)
        self.lock = threading.Lock()  # Thread-safe session management
    
    def authenticate(self, credentials: Dict[str, Any], session_id: str) -> bool:
        """Authenticate a user with email and OTP or token."""
        try:
            with self.lock:
                email = credentials.get('email')
                otp = credentials.get('otp')
                token = credentials.get('token')
                user_ip = credentials.get('ip')
                
                if email and otp:
                    if self.verify_otp(email, otp):
                        self.logger.info(f"Email OTP authentication successful for {email}")
                        if self.validate_license(email, user_ip, session_id):
                            self.redis.setex(f"session:{session_id}", 3600, f"{email}:{user_ip}")
                            return True
                    self.logger.error(f"Invalid OTP for {email}")
                    return False
                elif token:
                    if self.validate_license(token, user_ip, session_id):
                        decoded = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                        email = decoded.get('email')
                        self.redis.setex(f"session:{session_id}", 3600, f"{email}:{user_ip}")
                        return True
                    return False
                else:
                    self.logger.error("Missing required credentials")
                    return False
        
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
    
    def validate_license(self, license_key: str, user_ip: Optional[str], session_id: str) -> bool:
        """Validate a license key, checking IP and session limits."""
        try:
            with self.lock:
                # Check license server
                if '@' in license_key:  # Email-based license
                    response = requests.get(
                        f"{self.license_server_url}/validate",
                        params={'email': license_key},
                        timeout=5
                    )
                else:  # Token-based
                    decoded = jwt.decode(license_key, self.secret_key, algorithms=["HS256"])
                    email = decoded.get('email')
                    exp = decoded.get('exp')
                    if exp < time.time():
                        self.logger.error("License expired")
                        return False
                    response = requests.get(
                        f"{self.license_server_url}/validate",
                        params={'email': email, 'token': license_key},
                        timeout=5
                    )
                
                if response.status_code != 200:
                    self.logger.error(f"License server error: {response.text}")
                    return False
                
                license_data = response.json()
                if not license_data.get('valid', False):
                    self.logger.error("Invalid license")
                    return False
                
                # Check IP
                if user_ip and 'allowed_ips' in license_data:
                    if user_ip not in license_data['allowed_ips']:
                        self.logger.error(f"Unauthorized IP: {user_ip}")
                        return False
                
                # Check concurrent sessions
                max_sessions = license_data.get('max_sessions', 3)
                active_sessions = self.redis.keys(f"session:*:{email}:*")
                if len(active_sessions) >= max_sessions and session_id not in active_sessions:
                    self.logger.error(f"Max concurrent sessions ({max_sessions}) reached for {email}")
                    return False
                
                return True
        
        except Exception as e:
            self.logger.error(f"License validation failed: {e}")
            return False
    
    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify an OTP for the given email."""
        with self.lock:
            cached_otp = self.otp_cache.get(email)
            if cached_otp and cached_otp == otp:
                del self.otp_cache[email]
                return True
            return False
    
    def send_otp(self, email: str) -> bool:
        """Send an OTP to the user’s email."""
        try:
            with self.lock:
                # Rate-limit OTP requests (5 per minute)
                rate_key = f"otp_rate:{email}"
                if self.redis.exists(rate_key) and int(self.redis.get(rate_key)) >= 5:
                    self.logger.error(f"OTP rate limit exceeded for {email}")
                    return False
                
                otp = ''.join(random.choices('0123456789', k=6))
                self.otp_cache[email] = otp
                
                msg = MIMEText(f"Your OTP for backtester access is: {otp}")
                msg['Subject'] = 'Backtester Authentication OTP'
                msg['From'] = self.smtp_config['sender']
                msg['To'] = email
                
                with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                    server.starttls()
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                    server.sendmail(self.smtp_config['sender'], email, msg.as_string())
                
                # Update rate limit
                if not self.redis.exists(rate_key):
                    self.redis.setex(rate_key, 60, 1)
                else:
                    self.redis.incr(rate_key)
                
                self.logger.info(f"OTP sent to {email}")
                return True
        
        except Exception as e:
            self.logger.error(f"Failed to send OTP to {email}: {e}")
            return False
    
    def log_auth_attempt(self, credentials: Dict[str, Any], session_id: str, success: bool) -> None:
        """Log an authentication attempt."""
        email = credentials.get('email', 'unknown')
        ip = credentials.get('ip', 'unknown')
        status = "success" if success else "failure"
        self.logger.info(f"Auth attempt: email={email}, ip={ip}, session={session_id}, status={status}")
    
    def end_session(self, session_id: str) -> None:
        """End a user session."""
        try:
            with self.lock:
                if self.redis.exists(f"session:{session_id}"):
                    self.redis.delete(f"session:{session_id}")
                    self.logger.info(f"Session ended: {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to end session {session_id}: {e}")