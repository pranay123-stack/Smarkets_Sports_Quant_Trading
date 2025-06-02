from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AuthManagerInterface(ABC):
    """Abstract base class for authentication managers in sports betting systems."""
    
    def __init__(self):
        """Initialize the authentication manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any], session_id: str) -> bool:
        """Authenticate a user with provided credentials for a session.
        
        Args:
            credentials (Dict[str, Any]): Credentials (e.g., email, token, IP).
            session_id (str): Unique session identifier.
            
        Returns:
            bool: True if authenticated, False otherwise.
        """
        pass
    
    @abstractmethod
    def validate_license(self, license_key: str, user_ip: Optional[str], session_id: str) -> bool:
        """Validate a license key, checking IP and session limits.
        
        Args:
            license_key (str): License key or token.
            user_ip (Optional[str]): User's IP address.
            session_id (str): Session identifier.
            
        Returns:
            bool: True if license is valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def log_auth_attempt(self, credentials: Dict[str, Any], session_id: str, success: bool) -> None:
        """Log an authentication attempt.
        
        Args:
            credentials (Dict[str, Any]): Credentials used.
            session_id (str): Session identifier.
            success (bool): Whether authentication succeeded.
        """
        pass
    
    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """End a user session.
        
        Args:
            session_id (str): Session identifier.
        """
        pass