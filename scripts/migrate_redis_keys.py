#!/usr/bin/env python3
"""
Redis Key Migration Script for Multi-Clinic Support

This script migrates existing Redis session keys from the old format
to the new clinic-namespaced format.

OLD FORMAT: session:{session_id}:{state_type}
NEW FORMAT: clinic:{clinic_id}:session:{session_id}:{state_type}

Usage:
    python migrate_redis_keys.py --default-clinic-id <uuid> [--dry-run] [--redis-url <url>]

Example:
    python migrate_redis_keys.py --default-clinic-id "550e8400-e29b-41d4-a716-446655440000" --dry-run
"""

import argparse
import logging
import sys
from typing import Optional

try:
    import redis
except ImportError:
    print("Error: redis package not installed. Install with: pip install redis")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate_redis_keys(
    redis_client: redis.Redis,
    default_clinic_id: str,
    dry_run: bool = True
) -> dict:
    """
    Migrate existing session keys to clinic-namespaced format.

    Args:
        redis_client: Redis client instance
        default_clinic_id: Clinic ID to use for existing sessions
        dry_run: If True, only show what would be migrated without making changes

    Returns:
        dict with migration statistics
    """
    stats = {
        "total_keys_found": 0,
        "keys_migrated": 0,
        "keys_skipped": 0,
        "errors": 0
    }

    # Find all existing session keys with old format
    # Old format: session:{session_id}:{state_type}
    old_pattern = "session:*"

    logger.info(f"Scanning for keys matching pattern: {old_pattern}")
    logger.info(f"Default clinic_id for migration: {default_clinic_id}")
    logger.info(f"Dry run mode: {dry_run}")
    logger.info("-" * 60)

    for old_key in redis_client.scan_iter(old_pattern):
        old_key_str = old_key.decode('utf-8') if isinstance(old_key, bytes) else old_key
        stats["total_keys_found"] += 1

        try:
            # Parse old key format: session:{session_id}:{state_type}
            parts = old_key_str.split(":", 2)

            if len(parts) < 3:
                logger.warning(f"Skipping malformed key: {old_key_str}")
                stats["keys_skipped"] += 1
                continue

            # Check if already in new format (starts with "clinic:")
            if parts[0] == "clinic":
                logger.debug(f"Skipping already migrated key: {old_key_str}")
                stats["keys_skipped"] += 1
                continue

            # Extract session_id and state_type
            # Example: session:+971501234567:global_state
            prefix = parts[0]  # "session"
            session_id = parts[1]  # "+971501234567"
            state_type = parts[2]  # "global_state"

            if prefix != "session":
                logger.warning(f"Skipping non-session key: {old_key_str}")
                stats["keys_skipped"] += 1
                continue

            # Create new key with clinic namespace
            # New format: clinic:{clinic_id}:session:{session_id}:{state_type}
            new_key = f"clinic:{default_clinic_id}:session:{session_id}:{state_type}"

            # Get value and TTL
            value = redis_client.get(old_key)
            ttl = redis_client.ttl(old_key)

            if value is None:
                logger.warning(f"Key exists but value is None: {old_key_str}")
                stats["keys_skipped"] += 1
                continue

            if dry_run:
                logger.info(f"[DRY RUN] Would migrate: {old_key_str} -> {new_key} (TTL: {ttl}s)")
            else:
                # Copy value to new key with same TTL
                if ttl > 0:
                    redis_client.setex(new_key, ttl, value)
                else:
                    # No TTL or expired - set without TTL
                    redis_client.set(new_key, value)

                logger.info(f"Migrated: {old_key_str} -> {new_key} (TTL: {ttl}s)")

                # Note: We don't delete old keys immediately to allow rollback
                # Old keys can be deleted after verification period

            stats["keys_migrated"] += 1

        except Exception as e:
            logger.error(f"Error migrating key {old_key_str}: {e}")
            stats["errors"] += 1

    return stats


def cleanup_old_keys(
    redis_client: redis.Redis,
    dry_run: bool = True
) -> dict:
    """
    Delete old session keys after migration is verified.

    Only call this after verifying migration was successful.

    Args:
        redis_client: Redis client instance
        dry_run: If True, only show what would be deleted

    Returns:
        dict with cleanup statistics
    """
    stats = {
        "keys_found": 0,
        "keys_deleted": 0
    }

    old_pattern = "session:*"

    logger.info(f"Cleaning up old keys matching pattern: {old_pattern}")
    logger.info(f"Dry run mode: {dry_run}")
    logger.info("-" * 60)

    for old_key in redis_client.scan_iter(old_pattern):
        old_key_str = old_key.decode('utf-8') if isinstance(old_key, bytes) else old_key

        # Skip if it's actually a new format key (shouldn't happen but safety check)
        if old_key_str.startswith("clinic:"):
            continue

        stats["keys_found"] += 1

        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {old_key_str}")
        else:
            redis_client.delete(old_key)
            logger.info(f"Deleted: {old_key_str}")

        stats["keys_deleted"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Redis session keys to clinic-namespaced format"
    )
    parser.add_argument(
        "--default-clinic-id",
        required=True,
        help="Default clinic UUID to use for existing sessions"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL (default: redis://localhost:6379/0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete old keys after migration (use with caution!)"
    )

    args = parser.parse_args()

    # Connect to Redis
    try:
        redis_client = redis.from_url(args.redis_url)
        redis_client.ping()
        logger.info(f"Connected to Redis at {args.redis_url}")
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)

    # Run migration
    logger.info("=" * 60)
    logger.info("REDIS KEY MIGRATION FOR MULTI-CLINIC SUPPORT")
    logger.info("=" * 60)

    if args.cleanup:
        # Cleanup mode
        stats = cleanup_old_keys(redis_client, dry_run=args.dry_run)
        logger.info("=" * 60)
        logger.info("CLEANUP COMPLETE")
        logger.info(f"  Keys found: {stats['keys_found']}")
        logger.info(f"  Keys deleted: {stats['keys_deleted']}")
    else:
        # Migration mode
        stats = migrate_redis_keys(
            redis_client,
            args.default_clinic_id,
            dry_run=args.dry_run
        )

        logger.info("=" * 60)
        logger.info("MIGRATION COMPLETE")
        logger.info(f"  Total keys found: {stats['total_keys_found']}")
        logger.info(f"  Keys migrated: {stats['keys_migrated']}")
        logger.info(f"  Keys skipped: {stats['keys_skipped']}")
        logger.info(f"  Errors: {stats['errors']}")

        if args.dry_run:
            logger.info("")
            logger.info("This was a DRY RUN. No changes were made.")
            logger.info("Run without --dry-run to perform actual migration.")
        else:
            logger.info("")
            logger.info("Migration complete. Old keys are preserved for rollback.")
            logger.info("After verifying migration, run with --cleanup to remove old keys.")


if __name__ == "__main__":
    main()
