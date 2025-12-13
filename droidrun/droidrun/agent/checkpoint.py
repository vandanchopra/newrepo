"""
State Checkpointing for DroidRun Agent.

Production-ready checkpoint system for:
- Periodic state serialization to disk
- Resume from checkpoint on restart
- Corruption recovery with backup files
- Configurable checkpoint intervals
"""

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("droidrun.checkpoint")


@dataclass
class CheckpointConfig:
    """Configuration for state checkpointing."""

    enabled: bool = False
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_seconds: float = 30.0
    max_checkpoints: int = 5
    auto_resume: bool = True
    compress: bool = False


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    checkpoint_id: str
    created_at: str
    agent_state_hash: str
    step_count: int
    instruction: str
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages state checkpointing for DroidAgent.

    Features:
    - Periodic automatic checkpointing
    - Manual checkpoint creation
    - Checkpoint rotation with max limit
    - Resume from latest valid checkpoint
    - Corruption detection and recovery
    """

    def __init__(
        self,
        config: CheckpointConfig = None,
        checkpoint_dir: str = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
            checkpoint_dir: Override checkpoint directory from config
        """
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_checkpoint_time = 0.0
        self._checkpoint_count = 0

        # Ensure checkpoint directory exists
        if self.config.enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _generate_checkpoint_id(self) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_{timestamp}_{self._checkpoint_count:04d}"

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the full path for a checkpoint."""
        return self.checkpoint_dir / f"{checkpoint_id}.json"

    def _get_backup_path(self, checkpoint_id: str) -> Path:
        """Get the backup path for a checkpoint."""
        return self.checkpoint_dir / f"{checkpoint_id}.json.bak"

    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute a hash of the state for integrity checking."""
        import hashlib
        state_str = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    async def save_checkpoint(
        self,
        state: Dict[str, Any],
        instruction: str = "",
        step_count: int = 0,
    ) -> str:
        """
        Save a checkpoint of the current state.

        Args:
            state: State dictionary to save
            instruction: Current instruction/goal
            step_count: Current step count

        Returns:
            Checkpoint ID
        """
        checkpoint_id = self._generate_checkpoint_id()
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        backup_path = self._get_backup_path(checkpoint_id)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            created_at=datetime.utcnow().isoformat(),
            agent_state_hash=self._compute_state_hash(state),
            step_count=step_count,
            instruction=instruction,
        )

        # Prepare checkpoint data
        checkpoint_data = {
            "metadata": metadata.to_dict(),
            "state": state,
        }

        try:
            # Write to temp file first, then rename (atomic operation)
            temp_path = checkpoint_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            # Create backup of existing checkpoint if present
            if checkpoint_path.exists():
                shutil.copy2(checkpoint_path, backup_path)

            # Atomic rename
            temp_path.rename(checkpoint_path)

            self._checkpoint_count += 1
            self._last_checkpoint_time = time.time()

            logger.info(f"✅ Checkpoint saved: {checkpoint_id}")

            # Rotate old checkpoints
            await self._rotate_checkpoints()

            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def load_checkpoint(
        self,
        checkpoint_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to load, or None for latest

        Returns:
            Checkpoint state dictionary, or None if not found
        """
        if checkpoint_id is None:
            checkpoint_id = await self.get_latest_checkpoint_id()
            if checkpoint_id is None:
                logger.info("No checkpoints found")
                return None

        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        backup_path = self._get_backup_path(checkpoint_id)

        # Try primary file first, then backup
        for path in [checkpoint_path, backup_path]:
            if not path.exists():
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)

                # Verify integrity
                metadata = CheckpointMetadata.from_dict(checkpoint_data["metadata"])
                state = checkpoint_data["state"]
                computed_hash = self._compute_state_hash(state)

                if computed_hash != metadata.agent_state_hash:
                    logger.warning(
                        f"Checkpoint {checkpoint_id} hash mismatch, trying backup..."
                    )
                    continue

                logger.info(f"✅ Checkpoint loaded: {checkpoint_id}")
                return state

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

        logger.warning(f"Could not load checkpoint {checkpoint_id}")
        return None

    async def get_latest_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint."""
        checkpoints = await self.list_checkpoints()
        if not checkpoints:
            return None
        # Sort by creation time (embedded in filename)
        checkpoints.sort(reverse=True)
        return checkpoints[0]

    async def list_checkpoints(self) -> List[str]:
        """List all available checkpoint IDs."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_*.json"):
            if not path.name.endswith(".bak"):
                checkpoints.append(path.stem)

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        backup_path = self._get_backup_path(checkpoint_id)

        deleted = False
        for path in [checkpoint_path, backup_path]:
            if path.exists():
                path.unlink()
                deleted = True

        if deleted:
            logger.info(f"🗑️ Checkpoint deleted: {checkpoint_id}")

        return deleted

    async def _rotate_checkpoints(self):
        """Remove old checkpoints beyond the max limit."""
        checkpoints = await self.list_checkpoints()
        checkpoints.sort(reverse=True)  # Newest first

        while len(checkpoints) > self.config.max_checkpoints:
            old_checkpoint = checkpoints.pop()
            await self.delete_checkpoint(old_checkpoint)
            logger.debug(f"Rotated old checkpoint: {old_checkpoint}")

    async def start_auto_checkpoint(
        self,
        state_provider: callable,
        instruction: str = "",
    ):
        """
        Start automatic periodic checkpointing.

        Args:
            state_provider: Async callable that returns current state dict
            instruction: Current instruction/goal
        """
        if not self.config.enabled:
            return

        self._running = True

        async def checkpoint_loop():
            step_count = 0
            while self._running:
                try:
                    await asyncio.sleep(self.config.checkpoint_interval_seconds)
                    if not self._running:
                        break

                    state = await state_provider()
                    step_count += 1
                    await self.save_checkpoint(
                        state=state,
                        instruction=instruction,
                        step_count=step_count,
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Auto-checkpoint failed: {e}")

        self._checkpoint_task = asyncio.create_task(checkpoint_loop())
        logger.info(
            f"🔄 Auto-checkpointing started (interval: {self.config.checkpoint_interval_seconds}s)"
        )

    async def stop_auto_checkpoint(self):
        """Stop automatic checkpointing."""
        self._running = False
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
            self._checkpoint_task = None
            logger.info("⏹️ Auto-checkpointing stopped")

    async def clear_all_checkpoints(self) -> int:
        """
        Clear all checkpoints.

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = await self.list_checkpoints()
        for checkpoint_id in checkpoints:
            await self.delete_checkpoint(checkpoint_id)
        return len(checkpoints)

    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        return {
            "enabled": self.config.enabled,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_count": self._checkpoint_count,
            "last_checkpoint_time": self._last_checkpoint_time,
            "auto_checkpoint_running": self._running,
            "max_checkpoints": self.config.max_checkpoints,
            "checkpoint_interval_seconds": self.config.checkpoint_interval_seconds,
        }


# Convenience function for creating a checkpoint manager
def create_checkpoint_manager(
    enabled: bool = False,
    checkpoint_dir: str = "checkpoints",
    interval_seconds: float = 30.0,
    max_checkpoints: int = 5,
) -> CheckpointManager:
    """
    Create a checkpoint manager with common settings.

    Args:
        enabled: Whether checkpointing is enabled
        checkpoint_dir: Directory for checkpoint files
        interval_seconds: Auto-checkpoint interval
        max_checkpoints: Maximum number of checkpoints to keep

    Returns:
        Configured CheckpointManager instance
    """
    config = CheckpointConfig(
        enabled=enabled,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_seconds=interval_seconds,
        max_checkpoints=max_checkpoints,
    )
    return CheckpointManager(config=config)
