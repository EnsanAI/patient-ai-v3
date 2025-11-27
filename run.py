#!/usr/bin/env python
"""
Simple script to run the Patient AI Service.

Usage:
    python run.py              # Run in development mode
    python run.py --prod       # Run in production mode
    python run.py --port 8080  # Run on custom port
"""

import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description='Run Patient AI Service')
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the service on (default: 8000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--prod',
        action='store_true',
        help='Run in production mode (no auto-reload)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level (default: info)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🦷 Patient AI Service v2.0")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Mode: {'Production' if args.prod else 'Development'}")
    print(f"Log Level: {args.log_level.upper()}")
    print("=" * 60)
    print(f"\n📡 API: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws\n")
    print("=" * 60)
    print("\nPress CTRL+C to stop\n")

    try:
        uvicorn.run(
            "patient_ai_service.api.server:app",
            host=args.host,
            port=args.port,
            reload=not args.prod,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Service stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
