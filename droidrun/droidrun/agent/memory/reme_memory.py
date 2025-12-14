"""
ReMe: Remember Me, Refine Me - Dynamic Procedural Memory Framework.

Based on: "Remember Me, Refine Me: A Dynamic Procedural Memory Framework
          for Experience-Driven Agent Evolution" (arXiv:2512.10696)

This module provides procedural memory that:
- Stores "how-to" knowledge from task executions
- Dynamically refines memories (not passive append-only)
- Uses multi-faceted distillation for experience extraction
- Supports context-adaptive reuse
- Implements utility-based refinement

Key Concepts from ReMe:
- Multi-faceted distillation: Extract success patterns, failure triggers, comparative insights
- Context-adaptive reuse: Tailor historical insights to new contexts
- Utility-based refinement: Add valid memories, prune outdated ones

Three Phases:
1. Experience Acquisition - Distill actionable knowledge from trajectories
2. Experience Reuse - Retrieve and adapt relevant experiences
3. Experience Refinement - Optimize experience pool quality
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("droidrun.memory.reme")


class ExperienceType(Enum):
    """Types of experiences stored in ReMe."""
    SUCCESS_PATTERN = "success_pattern"  # What worked
    FAILURE_TRIGGER = "failure_trigger"  # What caused failures
    COMPARATIVE_INSIGHT = "comparative_insight"  # A vs B analysis
    PROCEDURAL_KNOWLEDGE = "procedural_knowledge"  # Step-by-step how-to
    CONTEXTUAL_ADAPTATION = "contextual_adaptation"  # Context-specific variation


@dataclass
class Experience:
    """
    A single experience entry in ReMe memory.

    Each experience captures actionable knowledge that can be
    reused in similar future situations.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ExperienceType = ExperienceType.PROCEDURAL_KNOWLEDGE

    # Core content
    description: str = ""  # Human-readable description
    knowledge: str = ""  # The actual procedural knowledge
    context: str = ""  # When this applies

    # Source information
    source_task: str = ""  # Original task that generated this
    source_trajectory: List[Dict[str, Any]] = field(default_factory=list)

    # Applicability
    preconditions: List[str] = field(default_factory=list)  # When to use
    postconditions: List[str] = field(default_factory=list)  # Expected outcomes
    app_context: List[str] = field(default_factory=list)  # Relevant apps

    # Quality metrics
    success_count: int = 0  # Times this led to success
    failure_count: int = 0  # Times this led to failure
    utility_score: float = 1.0  # Computed utility (higher = more valuable)
    confidence: float = 1.0  # Confidence in this knowledge

    # Temporal info
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    last_validated_at: Optional[datetime] = None

    # Embedding for retrieval
    embedding: Optional[List[float]] = None

    # Tags for organization
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "knowledge": self.knowledge,
            "context": self.context,
            "source_task": self.source_task,
            "source_trajectory": self.source_trajectory,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "app_context": self.app_context,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "utility_score": self.utility_score,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_validated_at": self.last_validated_at.isoformat() if self.last_validated_at else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        data = data.copy()
        data["type"] = ExperienceType(data["type"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_used_at"):
            data["last_used_at"] = datetime.fromisoformat(data["last_used_at"])
        if data.get("last_validated_at"):
            data["last_validated_at"] = datetime.fromisoformat(data["last_validated_at"])
        # Don't include embedding in reconstruction
        data.pop("embedding", None)
        return cls(**data)

    def compute_utility(self) -> float:
        """
        Compute utility score based on success/failure history.

        Utility = (success_count + 1) / (success_count + failure_count + 2) * confidence * recency
        """
        total = self.success_count + self.failure_count + 2  # Laplace smoothing
        base_utility = (self.success_count + 1) / total

        # Recency factor (decay over 30 days)
        if self.last_used_at:
            days_since_use = (datetime.utcnow() - self.last_used_at).days
            recency = max(0.5, 1.0 - (days_since_use / 60))
        else:
            recency = 0.8

        self.utility_score = base_utility * self.confidence * recency
        return self.utility_score


@dataclass
class Trajectory:
    """
    A trajectory is a sequence of states and actions from task execution.

    Used as input for experience distillation.
    """

    task: str
    goal: str
    steps: List[Dict[str, Any]]  # Each step has action, observation, success
    final_success: bool
    final_reason: str
    app_packages: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReMeConfig:
    """Configuration for ReMe memory system."""

    # Storage
    persist_path: str = ".droidrun/reme_memory.json"
    max_experiences: int = 5000

    # Distillation
    min_trajectory_steps: int = 2  # Min steps to distill
    max_experiences_per_trajectory: int = 5  # Max experiences from one trajectory

    # Retrieval
    default_top_k: int = 5
    similarity_threshold: float = 0.5

    # Refinement
    utility_threshold: float = 0.3  # Below this = candidate for pruning
    staleness_days: int = 30  # Days without use = stale
    min_validations: int = 3  # Min uses before considering reliable

    # Embedding
    embedding_dimension: int = 384

    # Auto-refinement
    auto_refine: bool = True
    refine_interval_hours: int = 24


class ReMeMemory:
    """
    ReMe: Dynamic Procedural Memory for Experience-Driven Agent Evolution.

    Implements the three-phase lifecycle:
    1. Experience Acquisition: Distill trajectories into experiences
    2. Experience Reuse: Retrieve and adapt experiences for new tasks
    3. Experience Refinement: Maintain high-quality experience pool
    """

    def __init__(
        self,
        config: Optional[ReMeConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.config = config or ReMeConfig()
        self._experiences: Dict[str, Experience] = {}
        self._embedding_fn = embedding_fn

        # Statistics
        self._distillation_count = 0
        self._retrieval_count = 0
        self._refinement_count = 0

        # Load persisted state
        self._load_state()

        # Background tasks
        self._refine_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background refinement task."""
        if self.config.auto_refine:
            self._refine_task = asyncio.create_task(self._refinement_loop())

    async def stop(self):
        """Stop background tasks and save state."""
        if self._refine_task:
            self._refine_task.cancel()
            try:
                await self._refine_task
            except asyncio.CancelledError:
                pass
        self._save_state()

    # =========================================================================
    # PHASE 1: Experience Acquisition (Distillation)
    # =========================================================================

    async def distill_trajectory(
        self,
        trajectory: Trajectory,
    ) -> List[Experience]:
        """
        Distill a trajectory into actionable experiences.

        Multi-faceted distillation extracts:
        1. Success patterns - What worked and why
        2. Failure triggers - What caused problems
        3. Comparative insights - What distinguished success from failure
        4. Procedural knowledge - Step-by-step procedures

        Args:
            trajectory: The execution trajectory to distill

        Returns:
            List of distilled experiences
        """
        if len(trajectory.steps) < self.config.min_trajectory_steps:
            return []

        experiences = []

        # Extract based on outcome
        if trajectory.final_success:
            # Success patterns
            exp = await self._distill_success_pattern(trajectory)
            if exp:
                experiences.append(exp)

            # Procedural knowledge (how-to)
            exp = await self._distill_procedural_knowledge(trajectory)
            if exp:
                experiences.append(exp)
        else:
            # Failure triggers
            exp = await self._distill_failure_trigger(trajectory)
            if exp:
                experiences.append(exp)

        # Comparative insights (works for both)
        insights = await self._distill_comparative_insights(trajectory)
        experiences.extend(insights)

        # Store all distilled experiences
        for exp in experiences:
            await self._store_experience(exp)

        self._distillation_count += 1
        logger.info(
            f"Distilled {len(experiences)} experiences from trajectory: "
            f"{trajectory.task[:50]}..."
        )

        return experiences

    async def _distill_success_pattern(
        self,
        trajectory: Trajectory,
    ) -> Optional[Experience]:
        """Extract what made this trajectory successful."""
        # Identify key successful actions
        successful_steps = [
            s for s in trajectory.steps
            if s.get("success", True)
        ]

        if not successful_steps:
            return None

        # Build pattern description
        key_actions = [s.get("action", "unknown") for s in successful_steps[-5:]]
        pattern_desc = f"To {trajectory.goal}: " + " -> ".join(key_actions)

        exp = Experience(
            type=ExperienceType.SUCCESS_PATTERN,
            description=f"Successful approach for: {trajectory.task[:100]}",
            knowledge=pattern_desc,
            context=f"Goal: {trajectory.goal}",
            source_task=trajectory.task,
            source_trajectory=trajectory.steps[-10:],  # Keep last 10 steps
            app_context=trajectory.app_packages,
            success_count=1,
            preconditions=self._extract_preconditions(trajectory),
            postconditions=[f"Task completed: {trajectory.final_reason[:100]}"],
            tags=["success", "pattern"] + trajectory.app_packages[:3],
        )

        return exp

    async def _distill_failure_trigger(
        self,
        trajectory: Trajectory,
    ) -> Optional[Experience]:
        """Extract what caused this trajectory to fail."""
        # Find where things went wrong
        failed_steps = [
            s for s in trajectory.steps
            if not s.get("success", True)
        ]

        # Build failure description
        if failed_steps:
            failure_desc = f"Failed action: {failed_steps[-1].get('action', 'unknown')}"
            error = failed_steps[-1].get("error", trajectory.final_reason)
        else:
            failure_desc = f"Task failed: {trajectory.final_reason}"
            error = trajectory.final_reason

        # What to avoid
        avoidance = f"Avoid: {failure_desc}. Error: {error[:200]}"

        exp = Experience(
            type=ExperienceType.FAILURE_TRIGGER,
            description=f"Failure case for: {trajectory.task[:100]}",
            knowledge=avoidance,
            context=f"Goal: {trajectory.goal}. This approach failed.",
            source_task=trajectory.task,
            source_trajectory=trajectory.steps[-5:],  # Keep context around failure
            app_context=trajectory.app_packages,
            failure_count=1,
            confidence=0.8,  # Slightly lower confidence for failure patterns
            preconditions=self._extract_preconditions(trajectory),
            postconditions=[f"AVOID: {error[:100]}"],
            tags=["failure", "trigger", "avoid"] + trajectory.app_packages[:3],
        )

        return exp

    async def _distill_procedural_knowledge(
        self,
        trajectory: Trajectory,
    ) -> Optional[Experience]:
        """Extract step-by-step procedural knowledge."""
        if len(trajectory.steps) < 3:
            return None

        # Build procedure
        steps_desc = []
        for i, step in enumerate(trajectory.steps[:10], 1):
            action = step.get("action", "unknown")
            steps_desc.append(f"{i}. {action}")

        procedure = "\n".join(steps_desc)

        exp = Experience(
            type=ExperienceType.PROCEDURAL_KNOWLEDGE,
            description=f"How to: {trajectory.goal[:100]}",
            knowledge=f"Procedure to {trajectory.goal}:\n{procedure}",
            context=f"Task: {trajectory.task}",
            source_task=trajectory.task,
            source_trajectory=trajectory.steps,
            app_context=trajectory.app_packages,
            success_count=1 if trajectory.final_success else 0,
            failure_count=0 if trajectory.final_success else 1,
            preconditions=self._extract_preconditions(trajectory),
            postconditions=[trajectory.final_reason[:100]],
            tags=["procedure", "how-to"] + trajectory.app_packages[:3],
        )

        return exp

    async def _distill_comparative_insights(
        self,
        trajectory: Trajectory,
    ) -> List[Experience]:
        """Extract insights by comparing with existing experiences."""
        insights = []

        # Find similar existing experiences
        similar = await self.retrieve(
            task=trajectory.task,
            context=trajectory.goal,
            top_k=3,
        )

        for exp, score in similar:
            if score < 0.7:  # Only compare fairly similar experiences
                continue

            # Compare outcomes
            current_success = trajectory.final_success
            past_success = exp.success_count > exp.failure_count

            if current_success != past_success:
                # Interesting! Different outcomes for similar tasks
                insight = Experience(
                    type=ExperienceType.COMPARATIVE_INSIGHT,
                    description=f"Insight: Different outcomes for similar tasks",
                    knowledge=(
                        f"Similar task '{exp.source_task[:50]}' "
                        f"{'succeeded' if past_success else 'failed'}, "
                        f"but '{trajectory.task[:50]}' "
                        f"{'succeeded' if current_success else 'failed'}. "
                        f"Key difference may be: {trajectory.final_reason[:100]}"
                    ),
                    context=f"Comparing similar tasks with different outcomes",
                    source_task=trajectory.task,
                    app_context=trajectory.app_packages,
                    success_count=1 if current_success else 0,
                    tags=["insight", "comparison"],
                )
                insights.append(insight)

        return insights[:2]  # Limit insights per trajectory

    def _extract_preconditions(self, trajectory: Trajectory) -> List[str]:
        """Extract preconditions from trajectory."""
        preconditions = []

        # First step often reveals preconditions
        if trajectory.steps:
            first_step = trajectory.steps[0]
            if "screen" in str(first_step).lower():
                preconditions.append("Screen must be on")
            if trajectory.app_packages:
                preconditions.append(f"App context: {trajectory.app_packages[0]}")

        return preconditions

    # =========================================================================
    # PHASE 2: Experience Reuse (Retrieval)
    # =========================================================================

    async def retrieve(
        self,
        task: str,
        context: str = "",
        top_k: Optional[int] = None,
        experience_types: Optional[List[ExperienceType]] = None,
        success_only: bool = False,
    ) -> List[Tuple[Experience, float]]:
        """
        Retrieve relevant experiences for a new task.

        Context-adaptive reuse tailors historical insights to new contexts
        via scenario-aware indexing.

        Args:
            task: The new task description
            context: Additional context (current state, goal, etc.)
            top_k: Number of experiences to retrieve
            experience_types: Filter by experience type
            success_only: Only return experiences with positive outcomes

        Returns:
            List of (experience, relevance_score) tuples
        """
        top_k = top_k or self.config.default_top_k
        query = f"{task} {context}"

        # Get query embedding
        if self._embedding_fn:
            query_embedding = self._embedding_fn(query)
        else:
            # Simple fallback: word overlap scoring
            query_embedding = None

        scored_experiences = []

        for exp in self._experiences.values():
            # Type filter
            if experience_types and exp.type not in experience_types:
                continue

            # Success filter
            if success_only and exp.success_count <= exp.failure_count:
                continue

            # Compute relevance score
            if query_embedding and exp.embedding:
                # Cosine similarity
                import numpy as np
                q = np.array(query_embedding)
                e = np.array(exp.embedding)
                score = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-8))
            else:
                # Fallback: keyword overlap
                score = self._keyword_similarity(query, exp)

            # Boost by utility
            score *= (0.5 + 0.5 * exp.utility_score)

            if score >= self.config.similarity_threshold:
                scored_experiences.append((exp, score))

        # Sort by score and return top_k
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        results = scored_experiences[:top_k]

        # Update last_used_at for retrieved experiences
        for exp, _ in results:
            exp.last_used_at = datetime.utcnow()

        self._retrieval_count += 1

        return results

    async def get_context_for_task(
        self,
        task: str,
        goal: str,
        max_context_length: int = 2000,
    ) -> str:
        """
        Get formatted context from experiences for a new task.

        This is the main interface for integrating ReMe into agent prompts.
        """
        # Retrieve relevant experiences
        experiences = await self.retrieve(
            task=task,
            context=goal,
            top_k=5,
        )

        if not experiences:
            return ""

        # Format for inclusion in prompt
        context_parts = ["## Relevant Past Experiences (ReMe)\n"]
        current_length = len(context_parts[0])

        # Group by type for better organization
        success_patterns = []
        failure_triggers = []
        procedures = []
        insights = []

        for exp, score in experiences:
            if exp.type == ExperienceType.SUCCESS_PATTERN:
                success_patterns.append((exp, score))
            elif exp.type == ExperienceType.FAILURE_TRIGGER:
                failure_triggers.append((exp, score))
            elif exp.type == ExperienceType.PROCEDURAL_KNOWLEDGE:
                procedures.append((exp, score))
            else:
                insights.append((exp, score))

        # Add success patterns
        if success_patterns:
            context_parts.append("\n### What Worked Before:\n")
            for exp, score in success_patterns[:2]:
                entry = f"- {exp.knowledge[:200]}\n"
                if current_length + len(entry) > max_context_length:
                    break
                context_parts.append(entry)
                current_length += len(entry)

        # Add failure triggers (what to avoid)
        if failure_triggers:
            context_parts.append("\n### What to Avoid:\n")
            for exp, score in failure_triggers[:2]:
                entry = f"- {exp.knowledge[:200]}\n"
                if current_length + len(entry) > max_context_length:
                    break
                context_parts.append(entry)
                current_length += len(entry)

        # Add procedures
        if procedures:
            context_parts.append("\n### Relevant Procedures:\n")
            for exp, score in procedures[:1]:
                entry = f"- {exp.knowledge[:300]}\n"
                if current_length + len(entry) > max_context_length:
                    break
                context_parts.append(entry)
                current_length += len(entry)

        return "".join(context_parts)

    def _keyword_similarity(self, query: str, exp: Experience) -> float:
        """Simple keyword-based similarity for fallback."""
        query_words = set(query.lower().split())
        exp_words = set(
            (exp.description + " " + exp.knowledge + " " + exp.context).lower().split()
        )

        if not query_words or not exp_words:
            return 0.0

        overlap = len(query_words & exp_words)
        return overlap / (len(query_words) + len(exp_words) - overlap + 1e-8)

    # =========================================================================
    # PHASE 3: Experience Refinement
    # =========================================================================

    async def refine(self) -> Dict[str, int]:
        """
        Refine the experience pool to maintain quality.

        Utility-based refinement:
        1. Prune low-utility experiences
        2. Merge similar experiences
        3. Update utility scores
        4. Remove stale experiences
        """
        stats = {
            "pruned": 0,
            "merged": 0,
            "updated": 0,
        }

        # Update all utility scores
        for exp in self._experiences.values():
            exp.compute_utility()
            stats["updated"] += 1

        # Prune low-utility experiences
        to_prune = []
        for exp_id, exp in self._experiences.items():
            # Low utility AND enough validations to be sure
            total_uses = exp.success_count + exp.failure_count
            if exp.utility_score < self.config.utility_threshold:
                if total_uses >= self.config.min_validations:
                    to_prune.append(exp_id)

            # Staleness check
            if exp.last_used_at:
                days_unused = (datetime.utcnow() - exp.last_used_at).days
                if days_unused > self.config.staleness_days:
                    # Reduce confidence of stale experiences
                    exp.confidence *= 0.9

        for exp_id in to_prune:
            del self._experiences[exp_id]
            stats["pruned"] += 1

        # Merge similar experiences
        stats["merged"] = await self._merge_similar_experiences()

        # Ensure we don't exceed max_experiences
        if len(self._experiences) > self.config.max_experiences:
            # Remove lowest utility experiences
            sorted_exps = sorted(
                self._experiences.items(),
                key=lambda x: x[1].utility_score,
            )
            excess = len(self._experiences) - self.config.max_experiences
            for exp_id, _ in sorted_exps[:excess]:
                del self._experiences[exp_id]
                stats["pruned"] += 1

        self._refinement_count += 1
        self._save_state()

        logger.info(
            f"ReMe refinement: pruned={stats['pruned']}, "
            f"merged={stats['merged']}, updated={stats['updated']}"
        )

        return stats

    async def _merge_similar_experiences(self) -> int:
        """Merge very similar experiences to reduce redundancy."""
        merged = 0
        to_remove = set()

        exp_list = list(self._experiences.values())

        for i, exp1 in enumerate(exp_list):
            if exp1.id in to_remove:
                continue

            for exp2 in exp_list[i + 1:]:
                if exp2.id in to_remove:
                    continue

                # Same type and high keyword similarity
                if exp1.type != exp2.type:
                    continue

                sim = self._keyword_similarity(
                    exp1.knowledge,
                    Experience(
                        description=exp2.description,
                        knowledge=exp2.knowledge,
                        context=exp2.context,
                    )
                )

                if sim > 0.8:  # Very similar
                    # Merge exp2 into exp1
                    exp1.success_count += exp2.success_count
                    exp1.failure_count += exp2.failure_count
                    exp1.confidence = (exp1.confidence + exp2.confidence) / 2

                    # Keep the better knowledge (longer usually means more detail)
                    if len(exp2.knowledge) > len(exp1.knowledge):
                        exp1.knowledge = exp2.knowledge

                    to_remove.add(exp2.id)
                    merged += 1

        for exp_id in to_remove:
            del self._experiences[exp_id]

        return merged

    async def update_experience_outcome(
        self,
        experience_id: str,
        success: bool,
    ):
        """
        Update an experience based on whether it led to success.

        This is called after using an experience to track its effectiveness.
        """
        if experience_id not in self._experiences:
            return

        exp = self._experiences[experience_id]
        if success:
            exp.success_count += 1
        else:
            exp.failure_count += 1

        exp.last_validated_at = datetime.utcnow()
        exp.compute_utility()

    async def _refinement_loop(self):
        """Background loop for periodic refinement."""
        while True:
            try:
                await asyncio.sleep(self.config.refine_interval_hours * 3600)
                await self.refine()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ReMe refinement error: {e}")

    # =========================================================================
    # Storage
    # =========================================================================

    async def _store_experience(self, exp: Experience):
        """Store a new experience."""
        # Generate embedding if possible
        if self._embedding_fn:
            text = f"{exp.description} {exp.knowledge} {exp.context}"
            exp.embedding = self._embedding_fn(text)

        self._experiences[exp.id] = exp

    def _save_state(self):
        """Save experiences to disk."""
        import os

        os.makedirs(os.path.dirname(self.config.persist_path), exist_ok=True)

        data = {
            "experiences": {
                exp_id: exp.to_dict()
                for exp_id, exp in self._experiences.items()
            },
            "stats": {
                "distillation_count": self._distillation_count,
                "retrieval_count": self._retrieval_count,
                "refinement_count": self._refinement_count,
            },
        }

        with open(self.config.persist_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load_state(self):
        """Load experiences from disk."""
        import os

        if not os.path.exists(self.config.persist_path):
            return

        try:
            with open(self.config.persist_path, 'r') as f:
                data = json.load(f)

            for exp_id, exp_data in data.get("experiences", {}).items():
                self._experiences[exp_id] = Experience.from_dict(exp_data)

            stats = data.get("stats", {})
            self._distillation_count = stats.get("distillation_count", 0)
            self._retrieval_count = stats.get("retrieval_count", 0)
            self._refinement_count = stats.get("refinement_count", 0)

            logger.info(
                f"Loaded ReMe memory: {len(self._experiences)} experiences"
            )
        except Exception as e:
            logger.warning(f"Failed to load ReMe memory: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get ReMe memory statistics."""
        type_counts = {}
        for exp in self._experiences.values():
            type_name = exp.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        total_success = sum(e.success_count for e in self._experiences.values())
        total_failure = sum(e.failure_count for e in self._experiences.values())

        return {
            "total_experiences": len(self._experiences),
            "by_type": type_counts,
            "total_success_uses": total_success,
            "total_failure_uses": total_failure,
            "distillation_count": self._distillation_count,
            "retrieval_count": self._retrieval_count,
            "refinement_count": self._refinement_count,
            "avg_utility": (
                sum(e.utility_score for e in self._experiences.values()) /
                max(1, len(self._experiences))
            ),
        }


# Convenience function
def create_reme_memory(
    persist_path: str = ".droidrun/reme_memory.json",
    embedding_fn: Optional[Callable] = None,
    **kwargs
) -> ReMeMemory:
    """Create a ReMe memory instance with common settings."""
    config = ReMeConfig(persist_path=persist_path, **kwargs)
    return ReMeMemory(config=config, embedding_fn=embedding_fn)
