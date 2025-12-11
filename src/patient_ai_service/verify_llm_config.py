#!/usr/bin/env python3
"""
Quick verification script to check LLM provider configuration.

Usage:
    python verify_llm_config.py

This script will:
1. Check current provider and model
2. Verify API key is set
3. Test LLM client creation
4. Run a simple test call (optional)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patient_ai_service.core.config import settings
from patient_ai_service.core.llm import get_llm_client, reset_llm_client
from patient_ai_service.models.enums import LLMProvider


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def verify_config():
    """Verify LLM configuration."""
    print_section("LLM Configuration Verification")
    
    # 1. Check provider
    print(f"\n✅ Provider: {settings.llm_provider.value}")
    print(f"   Type: {type(settings.llm_provider).__name__}")
    
    # 2. Check model
    try:
        model = settings.get_llm_model()
        print(f"✅ Model: {model}")
    except Exception as e:
        print(f"❌ Error getting model: {e}")
        return False
    
    # 3. Check API key
    try:
        api_key = settings.get_llm_api_key()
        if api_key:
            # Mask API key for security
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"✅ API Key: {masked_key} (set)")
        else:
            print(f"❌ API Key: NOT SET")
            print(f"   Please set {settings.llm_provider.value.upper()}_API_KEY environment variable")
            return False
    except Exception as e:
        print(f"❌ Error getting API key: {e}")
        return False
    
    # 4. Validate config
    try:
        settings.validate_llm_config()
        print(f"✅ Configuration validation: PASSED")
    except Exception as e:
        print(f"❌ Configuration validation: FAILED")
        print(f"   Error: {e}")
        return False
    
    return True


def test_client_creation():
    """Test LLM client creation."""
    print_section("Testing LLM Client Creation")
    
    try:
        # Reset to force recreation
        reset_llm_client()
        
        # Create client
        client = get_llm_client()
        client_type = type(client).__name__
        print(f"✅ Client created: {client_type}")
        
        # Check client attributes
        if hasattr(client, 'model'):
            print(f"✅ Client model: {client.model}")
        if hasattr(client, 'provider'):
            print(f"✅ Client provider: {client.provider}")
        
        return True, client
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_simple_call(client):
    """Test a simple LLM call."""
    print_section("Testing Simple LLM Call")
    
    try:
        print("Sending test message...")
        response = await client.create_message(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'Hello, I am working!' in one sentence."}],
            temperature=0.1,
            max_tokens=50
        )
        
        print(f"✅ Response received: {response[:100]}...")
        print(f"✅ Test call: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Test call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    print("\n" + "=" * 60)
    print("  LLM Provider Configuration Verification")
    print("=" * 60)
    
    # Step 1: Verify config
    if not verify_config():
        print("\n❌ Configuration verification failed. Please fix issues above.")
        sys.exit(1)
    
    # Step 2: Test client creation
    success, client = test_client_creation()
    if not success:
        print("\n❌ Client creation failed. Please check configuration.")
        sys.exit(1)
    
    # Step 3: Optional test call
    import asyncio
    print("\n" + "-" * 60)
    print("Optional: Test LLM call? (This will use API credits)")
    print("-" * 60)
    
    test_call = input("Run test call? (y/N): ").strip().lower()
    
    if test_call == 'y':
        try:
            result = asyncio.run(test_simple_call(client))
            if not result:
                print("\n⚠️  Test call failed, but client was created successfully.")
                print("   This might be a temporary API issue.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Test call cancelled by user.")
        except Exception as e:
            print(f"\n⚠️  Test call error: {e}")
    else:
        print("\n⏭️  Skipping test call.")
    
    # Summary
    print_section("Verification Summary")
    print(f"✅ Provider: {settings.llm_provider.value}")
    print(f"✅ Model: {settings.get_llm_model()}")
    print(f"✅ Client: {type(client).__name__}")
    print(f"\n✅ Configuration is valid and ready to use!")
    print(f"\n💡 Tip: Check logs on service startup to confirm provider/model")


if __name__ == "__main__":
    main()

