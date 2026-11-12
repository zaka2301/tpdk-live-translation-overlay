"""Event bus behavior: thread-safe delivery and slow-client isolation."""

from __future__ import annotations

import asyncio

import pytest

from tpdk_overlay.contracts import SessionState, SessionStateEvent
from tpdk_overlay.pipeline.events import EventBus, SubscriptionClosed


@pytest.mark.asyncio
async def test_publish_from_worker_thread_delivers() -> None:
    bus = EventBus()
    subscription = bus.subscribe()
    event = SessionStateEvent(state=SessionState.RUNNING)

    await asyncio.to_thread(bus.publish, event)

    received = await asyncio.wait_for(subscription.get(), timeout=1.0)
    assert received is event


@pytest.mark.asyncio
async def test_slow_subscriber_is_dropped_without_blocking_the_producer() -> None:
    bus = EventBus(max_queue_size=2)
    subscription = bus.subscribe()
    event = SessionStateEvent(state=SessionState.IDLE)

    for _ in range(10):
        bus.publish(event)
    await asyncio.sleep(0)

    received = 0
    with pytest.raises(SubscriptionClosed, match="outbound queue full"):
        while True:
            await subscription.get()
            received += 1
    assert received < 10
    assert subscription.closed


@pytest.mark.asyncio
async def test_unsubscribe_closes_a_pending_reader() -> None:
    bus = EventBus()
    subscription = bus.subscribe()
    waiter = asyncio.create_task(subscription.get())
    await asyncio.sleep(0)

    bus.unsubscribe(subscription)

    with pytest.raises(SubscriptionClosed):
        await waiter
    assert len(bus) == 0
