#!/usr/bin/env python3
"""
Log viewer utility for SecureRAG.
Makes it easy to view and extract logs for debugging.
"""

import sys
from pathlib import Path
import argparse


def get_log_path():
    """Get the path to the log file"""
    log_dir = Path.home() / ".secure-rag" / "logs"
    return log_dir / "secure-rag.log"


def tail_log(lines=50):
    """Show last N lines of log"""
    log_path = get_log_path()

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("The log file will be created when SecureRAG first runs.")
        return

    with open(log_path) as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())


def show_errors(lines=100):
    """Show only error and warning lines"""
    log_path = get_log_path()

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return

    with open(log_path) as f:
        for line in f:
            if any(level in line for level in ["ERROR", "WARNING", "CRITICAL"]):
                print(line.rstrip())


def follow_log():
    """Follow log in real-time (like tail -f)"""
    log_path = get_log_path()

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        print("Waiting for log file to be created...")

    try:
        import time
        last_size = 0

        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size

                if current_size > last_size:
                    with open(log_path) as f:
                        f.seek(last_size)
                        new_content = f.read()
                        print(new_content, end='')
                    last_size = current_size

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped following log.")


def show_path():
    """Show the log file path"""
    log_path = get_log_path()
    print(f"Log file location: {log_path}")

    if log_path.exists():
        size_mb = log_path.stat().st_size / (1024 * 1024)
        print(f"Log file size: {size_mb:.2f} MB")
        print(f"\nYou can view it with:")
        print(f"  cat {log_path}")
        print(f"  tail -f {log_path}")
        print(f"  less {log_path}")
    else:
        print("\nLog file does not exist yet.")
        print("It will be created when SecureRAG first runs.")


def clear_log():
    """Clear the log file"""
    log_path = get_log_path()

    if not log_path.exists():
        print("No log file to clear.")
        return

    response = input(f"Clear log file at {log_path}? [y/N]: ")
    if response.lower() == 'y':
        log_path.write_text("")
        print("Log file cleared.")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="SecureRAG Log Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show last 50 lines
  python scripts/view_logs.py

  # Show last 100 lines
  python scripts/view_logs.py --tail 100

  # Show only errors
  python scripts/view_logs.py --errors

  # Follow log in real-time
  python scripts/view_logs.py --follow

  # Show log file path
  python scripts/view_logs.py --path

  # Clear log file
  python scripts/view_logs.py --clear
"""
    )

    parser.add_argument(
        '--tail', '-t',
        type=int,
        metavar='N',
        help='Show last N lines (default: 50)'
    )

    parser.add_argument(
        '--errors', '-e',
        action='store_true',
        help='Show only errors and warnings'
    )

    parser.add_argument(
        '--follow', '-f',
        action='store_true',
        help='Follow log in real-time'
    )

    parser.add_argument(
        '--path', '-p',
        action='store_true',
        help='Show log file path'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear log file'
    )

    args = parser.parse_args()

    # Execute command
    if args.path:
        show_path()
    elif args.errors:
        show_errors()
    elif args.follow:
        follow_log()
    elif args.clear:
        clear_log()
    else:
        lines = args.tail if args.tail else 50
        tail_log(lines)


if __name__ == "__main__":
    main()
