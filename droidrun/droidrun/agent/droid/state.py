from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict


# Memory bounds to prevent OOM during long-running tasks
MAX_ACTION_HISTORY = 100  # Keep last N actions
MAX_SUMMARY_HISTORY = 50  # Keep last N summaries
MAX_ERROR_DESCRIPTIONS = 20  # Keep last N errors
MAX_MESSAGE_HISTORY = 30  # Keep last N messages
MAX_A11Y_TREE_NODES = 500  # Limit UI tree size
MAX_SCRIPTER_HISTORY = 20  # Keep last N scripts


class DroidAgentState(BaseModel):
    """
    State model for DroidAgent workflow - shared across parent and child workflows.

    Memory Management:
    - All list fields have maximum sizes defined above
    - Call prune_history() periodically to trim old entries
    - This prevents OOM during long-running autonomous tasks
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Task context
    instruction: str = ""
    step_number: int = 0
    # App Cards
    app_card: str = ""
    # Formatted device state for prompts (complete text)
    formatted_device_state: str = ""

    # Focused element text
    focused_text: str = ""

    # Raw device state components (for access to raw data)
    a11y_tree: List[Dict] = Field(default_factory=list)
    phone_state: Dict = Field(default_factory=dict)

    # Private fields
    current_package_name: str = ""
    current_activity_name: str = ""
    visited_packages: set = Field(default_factory=set)
    visited_activities: set = Field(default_factory=set)

    # Previous device state (for before/after comparison in Manager)
    previous_formatted_device_state: str = ""

    # Screen dimensions and screenshot
    width: int = 0
    height: int = 0
    screenshot: str | bytes | None = None

    # Text manipulation flag
    has_text_to_modify: bool = False

    # Action tracking
    action_pool: List[Dict] = Field(default_factory=list)
    action_history: List[Dict] = Field(default_factory=list)
    summary_history: List[str] = Field(default_factory=list)
    action_outcomes: List[bool] = Field(default_factory=list)  # "A", "B", "C"
    error_descriptions: List[str] = Field(default_factory=list)

    # Last action info
    last_action: Dict = Field(default_factory=dict)
    last_summary: str = ""
    last_action_thought: str = ""

    # Memory
    memory: str = ""
    memory_context: str = ""  # Recalled episodes from memory system
    message_history: List[Dict] = Field(default_factory=list)

    # Planning
    plan: str = ""
    current_subgoal: str = ""
    finish_thought: str = ""
    progress_status: str = ""
    manager_answer: str = ""  # For answer-type tasks

    # Error handling
    error_flag_plan: bool = False
    err_to_manager_thresh: int = 2
    user_id: str | None = None
    # Script execution tracking
    scripter_history: List[Dict] = Field(default_factory=list)
    last_scripter_message: str = ""
    last_scripter_success: bool = True

    text_manipulation_history: List[Dict] = Field(default_factory=list)
    last_text_manipulation_success: bool = False

    output_dir: str = ""

    # Custom variables (user-defined)
    custom_variables: Dict = Field(default_factory=dict)

    def update_current_app(self, package_name: str, activity_name: str):
        """
        Update package and activity together, capturing telemetry event only once.

        This prevents duplicate PackageVisitEvents when both package and activity change.
        """
        # Check if either changed
        package_changed = package_name != self.current_package_name
        activity_changed = activity_name != self.current_activity_name

        if not (package_changed or activity_changed):
            return  # No change, nothing to do

        # Update tracking sets
        if package_changed and package_name:
            self.visited_packages.add(package_name)
        if activity_changed and activity_name:
            self.visited_activities.add(activity_name)

        # Update values
        self.current_package_name = package_name
        self.current_activity_name = activity_name

        # Capture telemetry event for any change
        # This ensures we track when apps close or transitions to empty state occur
        from droidrun.telemetry import PackageVisitEvent, capture

        capture(
            PackageVisitEvent(
                package_name=package_name or "Unknown",
                activity_name=activity_name or "Unknown",
                step_number=self.step_number,
            ),
            user_id=self.user_id,
        )

    def prune_history(self) -> Dict[str, int]:
        """
        Prune old entries from history lists to prevent memory growth.

        Should be called periodically during long-running tasks
        (e.g., every 10 steps or when memory usage is high).

        Returns:
            Dictionary with counts of pruned items per field
        """
        pruned = {}

        # Prune action history
        if len(self.action_history) > MAX_ACTION_HISTORY:
            excess = len(self.action_history) - MAX_ACTION_HISTORY
            self.action_history = self.action_history[-MAX_ACTION_HISTORY:]
            pruned["action_history"] = excess

        # Prune action pool (recent actions for context)
        if len(self.action_pool) > MAX_ACTION_HISTORY:
            excess = len(self.action_pool) - MAX_ACTION_HISTORY
            self.action_pool = self.action_pool[-MAX_ACTION_HISTORY:]
            pruned["action_pool"] = excess

        # Prune summary history
        if len(self.summary_history) > MAX_SUMMARY_HISTORY:
            excess = len(self.summary_history) - MAX_SUMMARY_HISTORY
            self.summary_history = self.summary_history[-MAX_SUMMARY_HISTORY:]
            pruned["summary_history"] = excess

        # Prune error descriptions
        if len(self.error_descriptions) > MAX_ERROR_DESCRIPTIONS:
            excess = len(self.error_descriptions) - MAX_ERROR_DESCRIPTIONS
            self.error_descriptions = self.error_descriptions[-MAX_ERROR_DESCRIPTIONS:]
            pruned["error_descriptions"] = excess

        # Prune message history
        if len(self.message_history) > MAX_MESSAGE_HISTORY:
            excess = len(self.message_history) - MAX_MESSAGE_HISTORY
            self.message_history = self.message_history[-MAX_MESSAGE_HISTORY:]
            pruned["message_history"] = excess

        # Prune action outcomes
        if len(self.action_outcomes) > MAX_ACTION_HISTORY:
            excess = len(self.action_outcomes) - MAX_ACTION_HISTORY
            self.action_outcomes = self.action_outcomes[-MAX_ACTION_HISTORY:]
            pruned["action_outcomes"] = excess

        # Prune scripter history
        if len(self.scripter_history) > MAX_SCRIPTER_HISTORY:
            excess = len(self.scripter_history) - MAX_SCRIPTER_HISTORY
            self.scripter_history = self.scripter_history[-MAX_SCRIPTER_HISTORY:]
            pruned["scripter_history"] = excess

        # Prune text manipulation history
        if len(self.text_manipulation_history) > MAX_SCRIPTER_HISTORY:
            excess = len(self.text_manipulation_history) - MAX_SCRIPTER_HISTORY
            self.text_manipulation_history = self.text_manipulation_history[-MAX_SCRIPTER_HISTORY:]
            pruned["text_manipulation_history"] = excess

        # Limit a11y_tree size (truncate deeply nested trees)
        if len(self.a11y_tree) > MAX_A11Y_TREE_NODES:
            self.a11y_tree = self.a11y_tree[:MAX_A11Y_TREE_NODES]
            pruned["a11y_tree"] = len(self.a11y_tree) - MAX_A11Y_TREE_NODES

        return pruned

    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with sizes of all tracked collections
        """
        return {
            "action_history": len(self.action_history),
            "action_pool": len(self.action_pool),
            "summary_history": len(self.summary_history),
            "error_descriptions": len(self.error_descriptions),
            "message_history": len(self.message_history),
            "action_outcomes": len(self.action_outcomes),
            "scripter_history": len(self.scripter_history),
            "text_manipulation_history": len(self.text_manipulation_history),
            "a11y_tree": len(self.a11y_tree),
            "visited_packages": len(self.visited_packages),
            "visited_activities": len(self.visited_activities),
        }

    def clear_transient_state(self):
        """
        Clear transient state between tasks while preserving learning.

        Call this when starting a new task to free memory while
        keeping important context like visited apps.
        """
        # Clear action tracking
        self.action_pool.clear()
        self.action_history.clear()
        self.summary_history.clear()
        self.action_outcomes.clear()
        self.error_descriptions.clear()

        # Clear device state (will be refreshed)
        self.formatted_device_state = ""
        self.previous_formatted_device_state = ""
        self.a11y_tree.clear()
        self.phone_state.clear()
        self.screenshot = None

        # Clear planning state
        self.plan = ""
        self.current_subgoal = ""
        self.finish_thought = ""
        self.progress_status = ""
        self.manager_answer = ""

        # Clear script state
        self.scripter_history.clear()
        self.text_manipulation_history.clear()

        # Reset counters
        self.step_number = 0
        self.error_flag_plan = False

        # Keep: visited_packages, visited_activities, memory, memory_context
