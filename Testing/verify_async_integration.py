#!/usr/bin/env python3
"""
Verification script for MCP Integration - Async/Await Compliance
Checks that all db_client calls are properly awaited
"""

import re
import sys
from pathlib import Path

# Files to check
FILES_TO_CHECK = [
    "src/patient_ai_service/infrastructure/db_ops_client.py",
    "src/patient_ai_service/agents/general_assistant.py",
    "src/patient_ai_service/agents/registration.py",
    "src/patient_ai_service/agents/emergency_response.py",
    "src/patient_ai_service/core/orchestrator.py",
]

def check_async_methods(filepath):
    """Check that async methods properly use await for db_client calls."""
    errors = []
    warnings = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_method = None
    current_method_is_async = False
    in_async_method = False
    method_indent_level = 0
    
    for i, line in enumerate(lines, 1):
        # Detect method definitions
        if re.match(r'\s*(async\s+)?def\s+\w+\s*\(', line):
            current_method = re.search(r'def\s+(\w+)', line).group(1)
            current_method_is_async = 'async' in line
            method_indent_level = len(line) - len(line.lstrip())
            
            # Check if this is a tool method
            if current_method.startswith('tool_'):
                if not current_method_is_async:
                    # Tool methods should be async if they might use db_client
                    if 'db_client' in ''.join(lines[i:min(i+100, len(lines))]):
                        errors.append(
                            f"Line {i}: Tool method '{current_method}()' is not async but uses db_client"
                        )
        
        # Check for db_client calls
        if 'self.db_client.' in line and current_method:
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Extract the db_client call
            match = re.search(r'self\.db_client\.(\w+)', line)
            if match:
                method_name = match.group(1)
                
                # Check if it's awaited
                before_call = line[:match.start()]
                has_await = 'await' in before_call
                
                # Get context - is this in an async method?
                is_in_async = current_method_is_async
                
                # dbops methods should all be async now
                if not has_await and is_in_async:
                    # This is a potential issue - db_client methods should be awaited
                    errors.append(
                        f"Line {i}: `db_client.{method_name}()` called without `await` in async method `{current_method}()`"
                    )
                
                # If not in async method, flag it
                if not is_in_async and not has_await:
                    errors.append(
                        f"Line {i}: `db_client.{method_name}()` called in non-async method `{current_method}()` (must be async)"
                    )
    
    return errors, warnings

def main():
    """Run verification on all files."""
    root = Path(".")
    total_errors = 0
    total_warnings = 0
    
    print("=" * 80)
    print("MCP INTEGRATION - ASYNC/AWAIT VERIFICATION")
    print("=" * 80)
    print()
    
    for filepath in FILES_TO_CHECK:
        full_path = root / filepath
        
        if not full_path.exists():
            print(f"⚠️  SKIP: {filepath} (file not found)")
            continue
        
        print(f"📋 Checking: {filepath}")
        errors, warnings = check_async_methods(str(full_path))
        
        if errors:
            print(f"  ❌ Found {len(errors)} error(s):")
            for error in errors:
                print(f"     {error}")
            total_errors += len(errors)
        
        if warnings:
            print(f"  ⚠️  Found {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"     {warning}")
            total_warnings += len(warnings)
        
        if not errors and not warnings:
            print(f"  ✅ OK - Async/await compliance verified")
        
        print()
    
    print("=" * 80)
    print(f"SUMMARY: {total_errors} errors, {total_warnings} warnings")
    print("=" * 80)
    
    if total_errors > 0:
        print("\n❌ INTEGRATION CHECK FAILED - Fix errors above before testing")
        return 1
    else:
        print("\n✅ INTEGRATION CHECK PASSED - Ready to test!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
