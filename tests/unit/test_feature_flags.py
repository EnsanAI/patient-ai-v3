"""
Unit tests for entity migration feature flags.

Tests that feature flags can be toggled and read correctly.
"""

import os
import pytest
from patient_ai_service.core.config import Settings


class TestFeatureFlags:
    """Test entity migration feature flags."""
    
    def test_default_values_are_false(self):
        """Test that feature flags default to False."""
        settings = Settings()
        assert settings.use_delta_entities is False
        assert settings.use_agent_scoped_derived is False
    
    def test_can_read_from_environment(self):
        """Test that flags can be read from environment variables."""
        # Set environment variables
        os.environ["USE_DELTA_ENTITIES"] = "true"
        os.environ["USE_AGENT_SCOPED_DERIVED"] = "true"
        
        # Create new settings instance (reads from env)
        settings = Settings()
        
        # Verify flags are True
        assert settings.use_delta_entities is True
        assert settings.use_agent_scoped_derived is True
        
        # Cleanup
        del os.environ["USE_DELTA_ENTITIES"]
        del os.environ["USE_AGENT_SCOPED_DERIVED"]
    
    def test_can_toggle_programmatically(self):
        """Test that flags can be toggled programmatically."""
        settings = Settings()
        
        # Initially False
        assert settings.use_delta_entities is False
        
        # Toggle to True
        settings.use_delta_entities = True
        assert settings.use_delta_entities is True
        
        # Toggle back to False
        settings.use_delta_entities = False
        assert settings.use_delta_entities is False
    
    def test_environment_variable_case_insensitive(self):
        """Test that environment variable names are case-insensitive."""
        # Set lowercase
        os.environ["use_delta_entities"] = "true"
        os.environ["use_agent_scoped_derived"] = "true"
        
        settings = Settings()
        
        # Should still work (Pydantic handles case-insensitive)
        assert settings.use_delta_entities is True
        assert settings.use_agent_scoped_derived is True
        
        # Cleanup
        del os.environ["use_delta_entities"]
        del os.environ["use_agent_scoped_derived"]
    
    def test_boolean_string_parsing(self):
        """Test that boolean strings are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
        ]
        
        for env_value, expected in test_cases:
            os.environ["USE_DELTA_ENTITIES"] = env_value
            settings = Settings()
            assert settings.use_delta_entities == expected, f"Failed for value: {env_value}"
            del os.environ["USE_DELTA_ENTITIES"]
    
    def test_flags_independent(self):
        """Test that flags can be set independently."""
        settings = Settings()
        
        # Set one flag
        settings.use_delta_entities = True
        assert settings.use_delta_entities is True
        assert settings.use_agent_scoped_derived is False  # Still False
        
        # Set other flag
        settings.use_agent_scoped_derived = True
        assert settings.use_delta_entities is True  # Still True
        assert settings.use_agent_scoped_derived is True

