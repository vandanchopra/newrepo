"""
App-specific integrations for DroidRun.

Provides specialized handlers for common apps:
- Termux: Shell command execution
- Browser: Web navigation and interaction
"""

from typing import Dict, Type

# App handlers registry
APP_HANDLERS: Dict[str, Type] = {}


def register_app_handler(package_name: str):
    """Decorator to register an app handler."""
    def decorator(cls):
        APP_HANDLERS[package_name] = cls
        return cls
    return decorator


def get_app_handler(package_name: str):
    """Get the handler for a specific app package."""
    return APP_HANDLERS.get(package_name)


def list_supported_apps():
    """List all supported app packages."""
    return list(APP_HANDLERS.keys())
