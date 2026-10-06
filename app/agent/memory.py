"""Bounded, process-local memory of user-visible Agent exchanges."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock


MAX_MESSAGES_PER_SESSION = 6
MAX_SESSIONS = 100


@dataclass
class _SessionGuardEntry:
    lock: object = field(default_factory=Lock)
    users: int = 0


class SessionMemoryStore:
    # Client-supplied IDs fit this single-machine project. Multi-user deployments
    # need high-entropy server-issued tokens or IDs bound to authenticated users.
    def __init__(self, *, max_messages=MAX_MESSAGES_PER_SESSION, max_sessions=MAX_SESSIONS):
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._lock = Lock()
        self._guards: dict[str, _SessionGuardEntry] = {}

    @contextmanager
    def session_guard(self, session_id: str):
        """Serialize one session's read → Agent → append transaction."""
        with self._lock:
            entry = self._guards.get(session_id)
            if entry is None:
                entry = _SessionGuardEntry()
                self._guards[session_id] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._lock:
                entry.users -= 1
                if entry.users == 0:
                    del self._guards[session_id]

    def get(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            messages = self._sessions.get(session_id)
            if messages is None:
                return []
            self._sessions.move_to_end(session_id)
            return [message.copy() for message in messages]

    def append_exchange(self, session_id: str, question: str, visible_answer: str) -> None:
        """Store only the current user text and the response actually shown to them."""
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.extend((
                {"role": "user", "content": question},
                {"role": "assistant", "content": visible_answer},
            ))
            del messages[:-self.max_messages]
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
