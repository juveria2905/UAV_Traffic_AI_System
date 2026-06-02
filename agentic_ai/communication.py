"""
================================================================================
agentic_ai/communication.py — Inter-Agent Message Bus
================================================================================

PURPOSE:
  Provides a lightweight publish/subscribe message bus that allows agents
  at different hierarchy levels to communicate without direct coupling.

WHY THIS FILE EXISTS:
  Without a communication layer, agents call each other's methods directly.
  This creates tight coupling: if MonitorAgent changes its API, every caller
  breaks. A message bus decouples senders from receivers — agents publish
  messages; subscribers react to message types they care about.

WHAT BREAKS IF REMOVED:
  - Agents cannot communicate asynchronously
  - No message history for debugging
  - No priority-based message routing
  - Decision tracing becomes impossible

DESIGN PATTERNS:
  - Publisher/Subscriber (Observer pattern, decoupled)
  - Priority Queue (urgent alerts delivered first)
  - Message Store (history for replay and debugging)

CONNECTS TO:
  - agent_system.py   → MonitorAgent, CoordinatorAgent, ExecutorAgent publish/subscribe
  - hierarchy_manager.py → ZoneAgent, CityAgent use routing
  - api.py            → REST endpoint exposes message history
  - reasoning_chain.py → decision messages linked to reasoning steps
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import threading

from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# MESSAGE PRIORITY LEVELS
# ==============================================================================

class MessagePriority(Enum):
    """
    Message priority determines delivery order in the queue.
    CRITICAL messages jump the queue ahead of NORMAL messages.

    Analogous to hospital triage — not first-come-first-served,
    but most-urgent-first.
    """
    CRITICAL = 0   # Emergency stop, imminent collision
    HIGH     = 1   # Reroute commands, high-risk alerts
    NORMAL   = 2   # Regular status updates, tracking data
    LOW      = 3   # Analytics, performance metrics, heartbeats


# ==============================================================================
# MESSAGE DATA STRUCTURE
# ==============================================================================

@dataclass
class AgentMessage:
    """
    A single message passed between agents.

    Fields:
        message_id:   Unique UUID for tracing this message through the system
        topic:        Routing key — subscribers filter by topic
                      Examples: "collision.high", "track.update", "decision.emergency"
        sender:       Agent name that published this message
        payload:      Arbitrary dict — the actual data being communicated
        priority:     Delivery priority (CRITICAL arrives before NORMAL)
        timestamp:    Unix timestamp when message was created
        target:       Optional specific recipient (None = broadcast to all subscribers)
        ttl:          Time-to-live in seconds (message expires after this)
                      Prevents stale collision alerts from being acted on late.

    EXAMPLE:
        AgentMessage(
            topic="collision.high",
            sender="MonitorAgent-Zone-A",
            payload={"track_id_a": 7, "track_id_b": 12, "ttc": 1.4},
            priority=MessagePriority.CRITICAL,
        )
    """
    topic:      str
    sender:     str
    payload:    Dict[str, Any]
    priority:   MessagePriority        = MessagePriority.NORMAL
    target:     Optional[str]          = None    # None = broadcast
    ttl:        float                  = 30.0    # seconds
    message_id: str                    = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:  float                  = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """True if this message is older than its TTL."""
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age_ms(self) -> float:
        """Age of this message in milliseconds."""
        return (time.time() - self.timestamp) * 1000

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "topic":      self.topic,
            "sender":     self.sender,
            "payload":    self.payload,
            "priority":   self.priority.name,
            "target":     self.target,
            "timestamp":  round(self.timestamp, 3),
            "age_ms":     round(self.age_ms, 1),
            "is_expired": self.is_expired,
        }

    def __lt__(self, other: "AgentMessage") -> bool:
        """Priority queue ordering: lower priority value = higher urgency."""
        return self.priority.value < other.priority.value


# ==============================================================================
# SUBSCRIPTION REGISTRY
# ==============================================================================

@dataclass
class Subscription:
    """
    A subscription: agent X wants to receive messages on topic Y.

    topic_pattern:
        Can be exact ("collision.high") or prefix-wildcard ("collision.*").
        The bus matches by checking if the message topic starts with the pattern
        after removing the ".*" suffix.

    handler:
        Callable invoked when a matching message arrives.
        Signature: handler(message: AgentMessage) -> None

    subscriber_name:
        Identifying label for the subscriber (used in logs and message history).
    """
    topic_pattern:   str
    handler:         Callable[[AgentMessage], None]
    subscriber_name: str
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def matches(self, topic: str) -> bool:
        """
        Check if a message topic matches this subscription's pattern.

        Examples:
            pattern "collision.*"  matches "collision.high", "collision.medium"
            pattern "collision.high" matches ONLY "collision.high"
            pattern "track.*" matches "track.update", "track.new"
        """
        if self.topic_pattern.endswith(".*"):
            prefix = self.topic_pattern[:-2]
            return topic.startswith(prefix)
        return self.topic_pattern == topic


# ==============================================================================
# MESSAGE BUS (THREAD-SAFE)
# ==============================================================================

class AgentMessageBus:
    """
    Thread-safe synchronous message bus for inter-agent communication.

    WHY NOT ASYNC?
    The pipeline runs synchronously (frame-by-frame in a loop).
    Using asyncio would require restructuring main.py substantially.
    Instead, we use threading.Lock for thread safety and process messages
    synchronously within each frame. This is simpler and just as effective
    at the 20fps processing rate we target.

    USAGE:
        bus = AgentMessageBus()

        # Subscribe (usually done in __init__ of each agent)
        bus.subscribe("collision.*", my_handler, "CoordinatorAgent")

        # Publish (from any agent, any thread)
        bus.publish(AgentMessage(
            topic="collision.high",
            sender="MonitorAgent-A",
            payload={"track_id_a": 7},
            priority=MessagePriority.CRITICAL,
        ))

        # Deliver all pending messages (call once per frame in main loop)
        bus.flush()
    """

    def __init__(
        self,
        history_maxlen:  int   = 500,
        enable_logging:  bool  = True,
        persist_to_file: bool  = False,
        persist_path:    Optional[Path] = None,
    ):
        """
        Args:
            history_maxlen:  Maximum messages kept in history (rolling buffer).
            enable_logging:  Whether to log each published message at DEBUG level.
            persist_to_file: Whether to append messages to a JSONL file on disk.
            persist_path:    Path for persistence file (required if persist_to_file).
        """
        self._lock:          threading.RLock      = threading.RLock()
        self._subscriptions: List[Subscription]   = []
        self._queue:         List[AgentMessage]   = []   # pending delivery
        self._history:       deque[AgentMessage]  = deque(maxlen=history_maxlen)
        self._message_count: Dict[str, int]       = defaultdict(int)   # per-topic counts
        self._enable_logging  = enable_logging
        self._persist         = persist_to_file
        self._persist_path    = persist_path
        self._total_published = 0
        self._total_delivered = 0

        if self._persist and self._persist_path:
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("AgentMessageBus initialized.")

    # ── SUBSCRIPTION MANAGEMENT ──────────────────────────────────────────────

    def subscribe(
        self,
        topic_pattern:   str,
        handler:         Callable[[AgentMessage], None],
        subscriber_name: str,
    ) -> str:
        """
        Register a handler for messages matching topic_pattern.

        Returns subscription_id for later unsubscription.

        THREAD SAFETY: acquires lock — safe to call from any thread.
        """
        sub = Subscription(
            topic_pattern=topic_pattern,
            handler=handler,
            subscriber_name=subscriber_name,
        )
        with self._lock:
            self._subscriptions.append(sub)

        logger.debug(
            f"[BUS] {subscriber_name} subscribed to '{topic_pattern}' "
            f"(id={sub.subscription_id})"
        )
        return sub.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID. Returns True if found and removed."""
        with self._lock:
            before = len(self._subscriptions)
            self._subscriptions = [
                s for s in self._subscriptions
                if s.subscription_id != subscription_id
            ]
            removed = len(self._subscriptions) < before

        if removed:
            logger.debug(f"[BUS] Subscription {subscription_id} removed.")
        return removed

    # ── PUBLISHING ────────────────────────────────────────────────────────────

    def publish(self, message: AgentMessage) -> None:
        """
        Add a message to the delivery queue.

        Messages are not delivered immediately — they are queued and delivered
        in priority order when flush() is called. This ensures all messages
        for a frame are collected before any are delivered, preventing
        cascading decisions within the same frame.

        THREAD SAFETY: acquires lock.
        """
        with self._lock:
            # Insert in priority order (CRITICAL before HIGH before NORMAL)
            inserted = False
            for i, existing in enumerate(self._queue):
                if message.priority.value < existing.priority.value:
                    self._queue.insert(i, message)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(message)

            self._history.append(message)
            self._message_count[message.topic] += 1
            self._total_published += 1

        if self._enable_logging:
            logger.debug(
                f"[BUS] PUBLISH [{message.priority.name}] "
                f"{message.sender} → '{message.topic}' "
                f"(id={message.message_id})"
            )

        if self._persist and self._persist_path:
            self._persist_message(message)

    def broadcast(
        self,
        topic:   str,
        sender:  str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl:     float = 30.0,
    ) -> None:
        """
        Convenience method: publish a broadcast message.
        Equivalent to publish() with target=None.
        """
        self.publish(AgentMessage(
            topic=topic,
            sender=sender,
            payload=payload,
            priority=priority,
            target=None,
            ttl=ttl,
        ))

    def send_direct(
        self,
        topic:   str,
        sender:  str,
        target:  str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """
        Convenience method: publish a directed message to a specific agent.
        Only subscribers with matching subscriber_name AND topic will receive it.
        """
        self.publish(AgentMessage(
            topic=topic,
            sender=sender,
            payload=payload,
            priority=priority,
            target=target,
        ))

    # ── DELIVERY ──────────────────────────────────────────────────────────────

    def flush(self) -> int:
        """
        Deliver all queued messages to matching subscribers.
        Call once per frame in the main pipeline loop.

        Messages are delivered in priority order.
        Expired messages (age > TTL) are discarded without delivery.
        Messages with a specific target are only delivered to that subscriber.

        Returns:
            Number of messages successfully delivered.

        WHY FLUSH-BASED DELIVERY:
        If messages were delivered immediately on publish(), a CollisionEngine
        message could trigger a MonitorAgent decision in the same frame as
        the frame that generated the collision — before the Coordinator has
        seen all collisions. Batching delivery ensures consistent per-frame state.
        """
        with self._lock:
            queue_snapshot   = list(self._queue)
            subs_snapshot    = list(self._subscriptions)
            self._queue.clear()

        delivered = 0
        expired   = 0

        for message in queue_snapshot:
            if message.is_expired:
                logger.debug(
                    f"[BUS] EXPIRED message '{message.topic}' "
                    f"from {message.sender} (age={message.age_ms:.0f}ms)"
                )
                expired += 1
                continue

            for sub in subs_snapshot:
                # Skip if message is directed to a different agent
                if message.target and message.target != sub.subscriber_name:
                    continue

                if sub.matches(message.topic):
                    try:
                        sub.handler(message)
                        delivered += 1
                    except Exception as e:
                        logger.error(
                            f"[BUS] Handler error in {sub.subscriber_name} "
                            f"for topic '{message.topic}': {e}"
                        )

        with self._lock:
            self._total_delivered += delivered

        if queue_snapshot:
            logger.debug(
                f"[BUS] Flush: {len(queue_snapshot)} messages → "
                f"{delivered} delivered, {expired} expired"
            )

        return delivered

    # ── QUERYING & HISTORY ────────────────────────────────────────────────────

    def get_recent_messages(
        self,
        topic_filter:  Optional[str] = None,
        sender_filter: Optional[str] = None,
        max_count:     int           = 50,
        min_priority:  Optional[MessagePriority] = None,
    ) -> List[dict]:
        """
        Retrieve recent message history for dashboard/API display.

        Args:
            topic_filter:   Return only messages matching this topic/prefix
            sender_filter:  Return only messages from this sender
            max_count:      Maximum number of messages to return
            min_priority:   Return only messages at least this priority
                            (CRITICAL=0, HIGH=1, NORMAL=2, LOW=3)
                            min_priority=HIGH returns CRITICAL + HIGH

        Returns:
            List of message dicts, most recent first.
        """
        with self._lock:
            history = list(self._history)

        # Apply filters
        result = []
        for msg in reversed(history):
            if topic_filter:
                pattern = topic_filter.rstrip("*").rstrip(".")
                if not msg.topic.startswith(pattern):
                    continue
            if sender_filter and msg.sender != sender_filter:
                continue
            if min_priority and msg.priority.value > min_priority.value:
                continue
            result.append(msg.to_dict())
            if len(result) >= max_count:
                break

        return result

    def get_stats(self) -> dict:
        """Return message bus statistics for monitoring."""
        with self._lock:
            topic_counts = dict(self._message_count)
            sub_count    = len(self._subscriptions)
            queue_len    = len(self._queue)

        return {
            "total_published":  self._total_published,
            "total_delivered":  self._total_delivered,
            "queue_pending":    queue_len,
            "subscriber_count": sub_count,
            "topic_counts":     topic_counts,
            "history_size":     len(self._history),
        }

    def get_subscription_summary(self) -> List[dict]:
        """Return list of active subscriptions for dashboard display."""
        with self._lock:
            return [
                {
                    "subscriber": s.subscriber_name,
                    "topic":      s.topic_pattern,
                    "id":         s.subscription_id,
                }
                for s in self._subscriptions
            ]

    def clear_queue(self) -> None:
        """Emergency clear of the pending queue (e.g., on system reset)."""
        with self._lock:
            cleared = len(self._queue)
            self._queue.clear()
        logger.warning(f"[BUS] Queue cleared — {cleared} messages discarded.")

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def _persist_message(self, message: AgentMessage) -> None:
        """Append message to JSONL file for offline analysis."""
        try:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"[BUS] Failed to persist message: {e}")


# ==============================================================================
# STANDARD TOPIC CONSTANTS
# ==============================================================================

class Topics:
    """
    Centralized topic name constants.
    Using constants prevents typos: "colision.high" vs "collision.high".

    Naming convention: <domain>.<severity_or_action>
    """
    # Collision topics
    COLLISION_HIGH   = "collision.high"
    COLLISION_MEDIUM = "collision.medium"
    COLLISION_LOW    = "collision.low"
    COLLISION_ALL    = "collision.*"

    # Track lifecycle
    TRACK_NEW        = "track.new"
    TRACK_UPDATE     = "track.update"
    TRACK_LOST       = "track.lost"
    TRACK_ALL        = "track.*"

    # Agent decisions
    DECISION_EMERGENCY  = "decision.emergency_stop"
    DECISION_REROUTE    = "decision.reroute"
    DECISION_HOLD       = "decision.hold"
    DECISION_MONITOR    = "decision.monitor"
    DECISION_PRIORITIZE = "decision.prioritize"
    DECISION_ALL        = "decision.*"

    # System events
    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_STATS     = "system.stats"
    SYSTEM_ALERT     = "system.alert"

    # Zone management
    ZONE_CONGESTION  = "zone.congestion"
    ZONE_CLEAR       = "zone.clear"
    ZONE_REASSIGN    = "zone.reassign"


# ==============================================================================
# GLOBAL BUS SINGLETON
# ==============================================================================

_global_bus: Optional[AgentMessageBus] = None


def get_message_bus(
    history_maxlen: int = 500,
    persist: bool = False,
    persist_path: Optional[Path] = None,
) -> AgentMessageBus:
    """
    Get the global message bus singleton.
    Creates it on first call.

    WHY SINGLETON:
    All agents in the system share one bus. If each agent created its own bus,
    messages published by MonitorAgent would never reach CoordinatorAgent.
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = AgentMessageBus(
            history_maxlen=history_maxlen,
            persist_to_file=persist,
            persist_path=persist_path,
        )
    return _global_bus


def reset_message_bus() -> None:
    """Reset the global bus (useful in tests to start fresh)."""
    global _global_bus
    _global_bus = None