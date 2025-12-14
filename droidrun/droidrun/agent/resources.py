"""
Device Resource Management for DroidRun.

Monitors device resources to enable intelligent task scheduling:
- Battery level and charging status
- Storage availability
- Memory usage
- Network connectivity
- Thermal state

Enables:
- Pause tasks on low battery
- Resume when charging
- Throttle on thermal warnings
- Alert on resource constraints
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("droidrun.resources")


@dataclass
class DeviceResources:
    """Current device resource state."""

    # Battery
    battery_level: int = 100  # 0-100
    is_charging: bool = False
    battery_health: str = "unknown"

    # Storage (in MB)
    storage_total_mb: int = 0
    storage_available_mb: int = 0
    storage_percent_used: float = 0.0

    # Memory (in MB)
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    memory_percent_used: float = 0.0

    # Network
    wifi_connected: bool = False
    mobile_data_connected: bool = False
    network_type: str = "unknown"

    # Thermal
    thermal_status: str = "normal"  # normal, moderate, severe, critical

    # Timestamp
    last_updated: float = field(default_factory=time.time)

    def is_battery_low(self, threshold: int = 20) -> bool:
        """Check if battery is below threshold."""
        return self.battery_level < threshold and not self.is_charging

    def is_storage_low(self, threshold_mb: int = 500) -> bool:
        """Check if storage is below threshold."""
        return self.storage_available_mb < threshold_mb

    def is_memory_low(self, threshold_percent: float = 90.0) -> bool:
        """Check if memory usage is above threshold."""
        return self.memory_percent_used > threshold_percent

    def is_thermal_warning(self) -> bool:
        """Check if device has thermal warning."""
        return self.thermal_status in ("severe", "critical")

    def can_execute_tasks(self, config: "ResourceConfig" = None) -> tuple[bool, str]:
        """
        Check if device can execute tasks.

        Returns:
            Tuple of (can_execute, reason)
        """
        config = config or ResourceConfig()

        if self.is_battery_low(config.min_battery_percent):
            return False, f"Battery too low ({self.battery_level}%)"

        if self.is_storage_low(config.min_storage_mb):
            return False, f"Storage too low ({self.storage_available_mb}MB)"

        if self.is_thermal_warning():
            return False, f"Thermal warning ({self.thermal_status})"

        if config.require_network and not (self.wifi_connected or self.mobile_data_connected):
            return False, "No network connection"

        return True, "OK"


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    enabled: bool = True

    # Battery thresholds
    min_battery_percent: int = 20
    pause_on_low_battery: bool = True
    resume_on_charging: bool = True

    # Storage thresholds
    min_storage_mb: int = 500

    # Memory thresholds
    max_memory_percent: float = 90.0

    # Network requirements
    require_network: bool = False
    prefer_wifi: bool = True

    # Thermal management
    pause_on_thermal_warning: bool = True

    # Monitoring interval
    check_interval_seconds: float = 60.0

    # Callbacks (set at runtime)
    on_low_battery: Optional[Callable[[DeviceResources], None]] = None
    on_charging_started: Optional[Callable[[DeviceResources], None]] = None
    on_thermal_warning: Optional[Callable[[DeviceResources], None]] = None
    on_resources_ok: Optional[Callable[[DeviceResources], None]] = None


class ResourceMonitor:
    """
    Monitors device resources and manages task execution constraints.

    Features:
    - Periodic resource checking
    - Automatic pause/resume based on battery
    - Thermal throttling
    - Callback notifications
    """

    def __init__(
        self,
        config: ResourceConfig = None,
        tools_instance: Any = None,
    ):
        """
        Initialize resource monitor.

        Args:
            config: Resource configuration
            tools_instance: AdbTools instance for device queries
        """
        self.config = config or ResourceConfig()
        self.tools = tools_instance
        self._resources = DeviceResources()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._paused_for_resources = False
        self._last_battery_state = None

    async def start(self):
        """Start resource monitoring."""
        if not self.config.enabled or self._running:
            return

        self._running = True

        # Initial check
        await self.refresh_resources()

        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🔋 Resource monitoring started")

    async def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("🔋 Resource monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
                if not self._running:
                    break

                await self.refresh_resources()
                await self._check_thresholds()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Resource monitor error: {e}")

    async def refresh_resources(self) -> DeviceResources:
        """
        Refresh resource information from device.

        Returns:
            Updated DeviceResources
        """
        if not self.tools:
            return self._resources

        try:
            # Get battery info
            battery_info = await self._get_battery_info()
            self._resources.battery_level = battery_info.get("level", 100)
            self._resources.is_charging = battery_info.get("charging", False)
            self._resources.battery_health = battery_info.get("health", "unknown")

            # Get storage info
            storage_info = await self._get_storage_info()
            self._resources.storage_total_mb = storage_info.get("total_mb", 0)
            self._resources.storage_available_mb = storage_info.get("available_mb", 0)
            if self._resources.storage_total_mb > 0:
                self._resources.storage_percent_used = (
                    (self._resources.storage_total_mb - self._resources.storage_available_mb)
                    / self._resources.storage_total_mb * 100
                )

            # Get memory info
            memory_info = await self._get_memory_info()
            self._resources.memory_total_mb = memory_info.get("total_mb", 0)
            self._resources.memory_available_mb = memory_info.get("available_mb", 0)
            if self._resources.memory_total_mb > 0:
                self._resources.memory_percent_used = (
                    (self._resources.memory_total_mb - self._resources.memory_available_mb)
                    / self._resources.memory_total_mb * 100
                )

            # Get network info
            network_info = await self._get_network_info()
            self._resources.wifi_connected = network_info.get("wifi", False)
            self._resources.mobile_data_connected = network_info.get("mobile", False)
            self._resources.network_type = network_info.get("type", "unknown")

            # Get thermal info
            thermal_info = await self._get_thermal_info()
            self._resources.thermal_status = thermal_info.get("status", "normal")

            self._resources.last_updated = time.time()

            logger.debug(
                f"Resources: battery={self._resources.battery_level}% "
                f"charging={self._resources.is_charging} "
                f"storage={self._resources.storage_available_mb}MB"
            )

        except Exception as e:
            logger.warning(f"Failed to refresh resources: {e}")

        return self._resources

    async def _get_battery_info(self) -> Dict[str, Any]:
        """Get battery information via ADB."""
        try:
            result = await self.tools.device.shell("dumpsys battery")

            info = {"level": 100, "charging": False, "health": "unknown"}

            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("level:"):
                    info["level"] = int(line.split(":")[1].strip())
                elif line.startswith("status:"):
                    status = int(line.split(":")[1].strip())
                    # 2 = charging, 5 = full
                    info["charging"] = status in (2, 5)
                elif line.startswith("health:"):
                    health_codes = {
                        1: "unknown", 2: "good", 3: "overheat",
                        4: "dead", 5: "over_voltage", 6: "unspecified_failure",
                    }
                    health_code = int(line.split(":")[1].strip())
                    info["health"] = health_codes.get(health_code, "unknown")

            return info

        except Exception as e:
            logger.debug(f"Battery info error: {e}")
            return {"level": 100, "charging": False, "health": "unknown"}

    async def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information via ADB."""
        try:
            result = await self.tools.device.shell("df /data")

            # Parse df output
            lines = result.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    # Values in 1K blocks
                    total_kb = int(parts[1])
                    available_kb = int(parts[3])
                    return {
                        "total_mb": total_kb // 1024,
                        "available_mb": available_kb // 1024,
                    }

            return {"total_mb": 0, "available_mb": 0}

        except Exception as e:
            logger.debug(f"Storage info error: {e}")
            return {"total_mb": 0, "available_mb": 0}

    async def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information via ADB."""
        try:
            result = await self.tools.device.shell("cat /proc/meminfo")

            info = {"total_mb": 0, "available_mb": 0}

            for line in result.split("\n"):
                if line.startswith("MemTotal:"):
                    # Value in kB
                    kb = int(re.search(r"\d+", line).group())
                    info["total_mb"] = kb // 1024
                elif line.startswith("MemAvailable:"):
                    kb = int(re.search(r"\d+", line).group())
                    info["available_mb"] = kb // 1024

            return info

        except Exception as e:
            logger.debug(f"Memory info error: {e}")
            return {"total_mb": 0, "available_mb": 0}

    async def _get_network_info(self) -> Dict[str, Any]:
        """Get network information via ADB."""
        try:
            result = await self.tools.device.shell("dumpsys connectivity")

            info = {"wifi": False, "mobile": False, "type": "unknown"}

            if "WIFI" in result and "CONNECTED" in result:
                info["wifi"] = True
                info["type"] = "wifi"
            if "MOBILE" in result and "CONNECTED" in result:
                info["mobile"] = True
                if not info["wifi"]:
                    info["type"] = "mobile"

            return info

        except Exception as e:
            logger.debug(f"Network info error: {e}")
            return {"wifi": False, "mobile": False, "type": "unknown"}

    async def _get_thermal_info(self) -> Dict[str, Any]:
        """Get thermal information via ADB."""
        try:
            result = await self.tools.device.shell("dumpsys thermalservice")

            status = "normal"

            if "SEVERE" in result or "CRITICAL" in result:
                status = "severe"
            elif "MODERATE" in result:
                status = "moderate"

            return {"status": status}

        except Exception as e:
            logger.debug(f"Thermal info error: {e}")
            return {"status": "normal"}

    async def _check_thresholds(self):
        """Check resource thresholds and trigger callbacks."""
        resources = self._resources

        # Check battery state changes
        was_charging = self._last_battery_state
        is_charging_now = resources.is_charging

        if was_charging is False and is_charging_now is True:
            # Started charging
            logger.info("🔌 Device started charging")
            if self.config.on_charging_started:
                self.config.on_charging_started(resources)

            if self._paused_for_resources and self.config.resume_on_charging:
                self._paused_for_resources = False
                if self.config.on_resources_ok:
                    self.config.on_resources_ok(resources)

        self._last_battery_state = is_charging_now

        # Check if we should pause
        can_execute, reason = resources.can_execute_tasks(self.config)

        if not can_execute and not self._paused_for_resources:
            self._paused_for_resources = True
            logger.warning(f"⚠️ Pausing tasks: {reason}")

            if resources.is_battery_low(self.config.min_battery_percent):
                if self.config.on_low_battery:
                    self.config.on_low_battery(resources)

            if resources.is_thermal_warning():
                if self.config.on_thermal_warning:
                    self.config.on_thermal_warning(resources)

        elif can_execute and self._paused_for_resources:
            self._paused_for_resources = False
            logger.info("✅ Resources OK, resuming tasks")
            if self.config.on_resources_ok:
                self.config.on_resources_ok(resources)

    @property
    def resources(self) -> DeviceResources:
        """Get current resource state."""
        return self._resources

    @property
    def is_paused(self) -> bool:
        """Check if tasks are paused due to resources."""
        return self._paused_for_resources

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource monitor statistics."""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "paused_for_resources": self._paused_for_resources,
            "battery_level": self._resources.battery_level,
            "is_charging": self._resources.is_charging,
            "storage_available_mb": self._resources.storage_available_mb,
            "memory_percent_used": self._resources.memory_percent_used,
            "thermal_status": self._resources.thermal_status,
            "last_updated": self._resources.last_updated,
        }


# Human Escalation Queue
@dataclass
class EscalationItem:
    """An item requiring human attention."""

    escalation_id: str
    task_id: str
    reason: str
    context: Dict[str, Any]
    screenshot_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolution: Optional[str] = None


class HumanEscalationQueue:
    """
    Queue for tasks that need human intervention.

    Scenarios requiring escalation:
    - Captcha/verification challenges
    - Login required
    - Ambiguous decisions
    - Repeated failures
    - Permission requests
    """

    def __init__(self, storage_path: str = "escalation_queue.json"):
        """Initialize escalation queue."""
        self._queue: List[EscalationItem] = []
        self._storage_path = storage_path
        self._callbacks: List[Callable[[EscalationItem], None]] = []
        self._load_queue()

    def _generate_id(self) -> str:
        """Generate unique escalation ID."""
        import uuid
        return f"esc_{uuid.uuid4().hex[:12]}"

    def escalate(
        self,
        task_id: str,
        reason: str,
        context: Dict[str, Any] = None,
        screenshot_path: str = None,
    ) -> str:
        """
        Add an item to the escalation queue.

        Args:
            task_id: ID of the task requiring escalation
            reason: Why escalation is needed
            context: Additional context (device state, last actions, etc.)
            screenshot_path: Path to screenshot showing the issue

        Returns:
            Escalation ID
        """
        item = EscalationItem(
            escalation_id=self._generate_id(),
            task_id=task_id,
            reason=reason,
            context=context or {},
            screenshot_path=screenshot_path,
        )

        self._queue.append(item)
        self._save_queue()

        logger.warning(f"🚨 Escalation created: {item.escalation_id} - {reason}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(item)
            except Exception as e:
                logger.error(f"Escalation callback error: {e}")

        return item.escalation_id

    def resolve(
        self,
        escalation_id: str,
        resolution: str,
    ) -> bool:
        """
        Resolve an escalation item.

        Args:
            escalation_id: ID of escalation to resolve
            resolution: Description of how it was resolved

        Returns:
            True if resolved, False if not found
        """
        for item in self._queue:
            if item.escalation_id == escalation_id:
                item.resolved = True
                item.resolved_at = time.time()
                item.resolution = resolution
                self._save_queue()
                logger.info(f"✅ Escalation resolved: {escalation_id}")
                return True

        return False

    def get_pending(self) -> List[EscalationItem]:
        """Get all pending (unresolved) escalations."""
        return [item for item in self._queue if not item.resolved]

    def get_by_task(self, task_id: str) -> List[EscalationItem]:
        """Get all escalations for a specific task."""
        return [item for item in self._queue if item.task_id == task_id]

    def on_escalation(self, callback: Callable[[EscalationItem], None]):
        """Register callback for new escalations."""
        self._callbacks.append(callback)

    def _save_queue(self):
        """Save queue to disk."""
        import json
        try:
            data = [
                {
                    "escalation_id": item.escalation_id,
                    "task_id": item.task_id,
                    "reason": item.reason,
                    "context": item.context,
                    "screenshot_path": item.screenshot_path,
                    "created_at": item.created_at,
                    "resolved": item.resolved,
                    "resolved_at": item.resolved_at,
                    "resolution": item.resolution,
                }
                for item in self._queue
            ]
            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save escalation queue: {e}")

    def _load_queue(self):
        """Load queue from disk."""
        import json
        import os

        if not os.path.exists(self._storage_path):
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)

            self._queue = [
                EscalationItem(**item)
                for item in data
            ]
            logger.info(f"Loaded {len(self._queue)} escalations from disk")
        except Exception as e:
            logger.error(f"Failed to load escalation queue: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = len([i for i in self._queue if not i.resolved])
        resolved = len([i for i in self._queue if i.resolved])

        return {
            "total": len(self._queue),
            "pending": pending,
            "resolved": resolved,
        }


# Convenience functions
def create_resource_monitor(
    tools_instance: Any = None,
    min_battery: int = 20,
    check_interval: float = 60.0,
) -> ResourceMonitor:
    """Create a resource monitor with common settings."""
    config = ResourceConfig(
        enabled=True,
        min_battery_percent=min_battery,
        check_interval_seconds=check_interval,
    )
    return ResourceMonitor(config=config, tools_instance=tools_instance)


def create_escalation_queue(
    storage_path: str = "escalation_queue.json",
) -> HumanEscalationQueue:
    """Create an escalation queue."""
    return HumanEscalationQueue(storage_path=storage_path)
