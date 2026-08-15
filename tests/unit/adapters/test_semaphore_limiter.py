import asyncio
import contextlib

from readeverything.adapters.semaphore_limiter import SemaphoreLimiter
from readeverything.domain.capability import Capability


async def test_concurrency_is_bounded_by_the_configured_limit() -> None:
    """Asserted by observing peak in-flight count, not by timing.

    A timing assertion would be flaky and would not actually establish the
    bound — a slow machine passes it for the wrong reason.
    """
    limiter = SemaphoreLimiter({Capability.VISION: 2})
    peak = 0
    in_flight = 0

    async def worker() -> None:
        nonlocal peak, in_flight
        async with limiter.limit(Capability.VISION):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert peak == 2


async def test_an_unconfigured_capability_is_unbounded_not_zero() -> None:
    """The failure mode of a mistake here must be visible load, never silence.

    A capability defaulting to zero would deadlock, and a deadlock looks
    exactly like the hang this whole cycle exists to eliminate.
    """
    limiter = SemaphoreLimiter({Capability.VISION: 1})
    ran = False
    async with limiter.limit(Capability.FFMPEG):
        ran = True
    assert ran


async def test_different_capabilities_do_not_share_a_bound() -> None:
    """A vision endpoint tolerating four in flight and ffmpeg bounded by cores
    are different constraints; one global number starves one or floods the
    other."""
    limiter = SemaphoreLimiter({Capability.VISION: 1, Capability.FFMPEG: 1})
    vision_peak = 0
    vision_in_flight = 0
    ffmpeg_peak = 0
    ffmpeg_in_flight = 0

    async def vision_worker() -> None:
        nonlocal vision_peak, vision_in_flight
        async with limiter.limit(Capability.VISION):
            vision_in_flight += 1
            vision_peak = max(vision_peak, vision_in_flight)
            await asyncio.sleep(0)
            vision_in_flight -= 1

    async def ffmpeg_worker() -> None:
        nonlocal ffmpeg_peak, ffmpeg_in_flight
        async with limiter.limit(Capability.FFMPEG):
            ffmpeg_in_flight += 1
            ffmpeg_peak = max(ffmpeg_peak, ffmpeg_in_flight)
            await asyncio.sleep(0)
            ffmpeg_in_flight -= 1

    await asyncio.gather(
        vision_worker(),
        vision_worker(),
        ffmpeg_worker(),
        ffmpeg_worker(),
    )
    assert vision_peak == 1
    assert ffmpeg_peak == 1


async def test_the_limit_is_released_when_the_body_raises() -> None:
    """Otherwise one failure permanently narrows the pipeline."""
    limiter = SemaphoreLimiter({Capability.VISION: 1})
    with contextlib.suppress(RuntimeError):
        async with limiter.limit(Capability.VISION):
            raise RuntimeError("boom")
    async with limiter.limit(Capability.VISION):
        pass  # reaching here is the assertion


def test_one_limiter_still_bounds_on_a_second_event_loop() -> None:
    """A limiter is a caller's configuration, and configuration outlives a loop.

    Deliberately synchronous: it drives two `asyncio.run` calls itself, which
    is the reuse pattern that used to hand loop two a semaphore owned by loop
    one and raise `RuntimeError: ... is bound to a different event loop`
    under contention. Contention is required to see it — a semaphore binds a
    loop when it must wait, so the workers must actually queue.
    """
    limiter = SemaphoreLimiter({Capability.VISION: 2})

    async def run_batch() -> int:
        peak = 0
        in_flight = 0

        async def worker() -> None:
            nonlocal peak, in_flight
            async with limiter.limit(Capability.VISION):
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(10)))
        return peak

    first = asyncio.run(run_batch())
    second = asyncio.run(run_batch())
    # Not merely "did not raise": the second loop must be bounded too, or the
    # fix could have been a fresh unbounded pass-through.
    assert first == 2
    assert second == 2
