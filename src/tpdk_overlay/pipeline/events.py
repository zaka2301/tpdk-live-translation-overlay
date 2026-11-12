"""Async publish/subscribe bus between pipeline threads and WebSocket clients.

Every connected UI owns one bounded :class:`asyncio.Queue`. Producers may run on
the uvicorn event loop or on worker threads; :meth:`EventBus.publish` marshals
each event onto the subscriber's loop with ``call_soon_threadsafe``, so a
producer never blocks on a slow client. A subscriber whose queue stays full is
closed and the WebSocket handler disconnects it; a dead client cannot stall the
pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from tpdk_overlay.contracts import Event

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256

_CLOSED = object()


class SubscriptionClosed(ConnectionError):
    """The subscriber was closed locally or dropped for being too slow."""


class Subscription:
    """One client's bounded event queue.

    ``get()`` is awaited from the WebSocket sender task. ``close()`` may be
    called from any thread; it wakes a pending ``get()`` with
    :class:`SubscriptionClosed`.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, max_queue_size: int) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=max_queue_size)
        self._closed = threading.Event()
        self._reason = "closed"

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def close(self, reason: str = "closed") -> None:
        """Close the subscription from any thread; idempotent."""

        if self._closed.is_set():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._close_on_loop(reason)
        else:
            try:
                self._loop.call_soon_threadsafe(self._close_on_loop, reason)
            except RuntimeError:
                # The loop is already gone; nothing can consume the queue.
                self._closed.set()
                self._reason = reason

    def _enqueue(self, event: Event) -> None:
        """Append on the subscriber's loop; drop the client when the queue is full."""

        if self._closed.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("dropping slow WebSocket client: outbound queue is full")
            self._close_on_loop("outbound queue full")

    def _close_on_loop(self, reason: str) -> None:
        if self._closed.is_set():
            return
        self._reason = reason
        self._closed.set()
        # Make room for the sentinel: the oldest event is already stale.
        while True:
            try:
                self._queue.put_nowait(_CLOSED)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

    async def get(self) -> Event:
        """Wait for the next event; raise :class:`SubscriptionClosed` when closed."""

        item = await self._queue.get()
        if item is _CLOSED:
            raise SubscriptionClosed(self._reason)
        return item  # type: ignore[return-value]


class EventBus:
    """Fan-out of pipeline events to every connected client."""

    def __init__(self, *, max_queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._max_queue_size = max_queue_size
        self._lock = threading.Lock()
        self._subscriptions: set[Subscription] = set()

    def subscribe(self) -> Subscription:
        """Create a subscription bound to the running event loop."""

        subscription = Subscription(asyncio.get_running_loop(), self._max_queue_size)
        with self._lock:
            self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(subscription)
        subscription.close()

    def publish(self, event: Event) -> None:
        """Deliver an event from any thread without blocking the producer."""

        with self._lock:
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            if subscription.closed:
                self.unsubscribe(subscription)
                continue
            try:
                subscription._loop.call_soon_threadsafe(subscription._enqueue, event)
            except RuntimeError:
                logger.debug("event loop closed while publishing %s", event.type)
                self.unsubscribe(subscription)

    def close(self) -> None:
        """Close every subscription, waking all senders."""

        with self._lock:
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._subscriptions)


__all__ = [
    "DEFAULT_QUEUE_SIZE",
    "EventBus",
    "Subscription",
    "SubscriptionClosed",
]
