import asyncio
import random
import time


# ============================================================
# TASK 16 CONFIGURATION
# ============================================================

NODE_TIMEOUT_SECONDS = 1.0
GLOBAL_TIMEOUT_SECONDS = 3.0

MAX_ATTEMPTS = 4
INITIAL_INTERVAL = 0.1
MAX_INTERVAL = 0.5
JITTER = 0.05


# ============================================================
# RETRY WITH EXPONENTIAL BACKOFF + JITTER
# ============================================================

async def retry_with_backoff(
    operation,
    max_attempts=MAX_ATTEMPTS,
    initial_interval=INITIAL_INTERVAL,
    max_interval=MAX_INTERVAL,
    jitter=JITTER,
):
    attempts = 0
    delays = []

    while attempts < max_attempts:
        attempts += 1

        try:
            result = await operation()

            return {
                "success": True,
                "attempts": attempts,
                "delays": delays,
                "result": result,
            }

        except Exception as exc:
            if attempts >= max_attempts:
                return {
                    "success": False,
                    "attempts": attempts,
                    "delays": delays,
                    "error": str(exc),
                }

            exponential_delay = min(
                initial_interval * (2 ** (attempts - 1)),
                max_interval,
            )

            jitter_value = random.uniform(0, jitter)

            delay = min(
                exponential_delay + jitter_value,
                max_interval,
            )

            delays.append(delay)

            await asyncio.sleep(delay)


# ============================================================
# SIMULATED TRANSIENT FAILURE
# ============================================================

class TransientOperation:
    def __init__(self, failures_before_success=2):
        self.failures_before_success = failures_before_success
        self.calls = 0

    async def __call__(self):
        self.calls += 1

        if self.calls <= self.failures_before_success:
            raise RuntimeError(
                f"Simulated transient failure #{self.calls}"
            )

        return "Operation recovered successfully"


# ============================================================
# PER-NODE TIMEOUT
# ============================================================

async def slow_node():
    await asyncio.sleep(2.0)
    return "slow node completed"


async def run_node_with_timeout():
    try:
        result = await asyncio.wait_for(
            slow_node(),
            timeout=NODE_TIMEOUT_SECONDS,
        )

        return {
            "success": True,
            "result": result,
        }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "timeout": True,
            "error": "Per-node timeout exceeded",
        }


# ============================================================
# FAST NODE
# ============================================================

async def fast_node():
    await asyncio.sleep(0.1)
    return "fast node completed"


# ============================================================
# GLOBAL TIMEOUT WORKFLOW
# ============================================================

async def global_workflow():
    await fast_node()

    # This intentionally takes longer than the global timeout.
    await asyncio.sleep(5.0)

    return "workflow completed"


async def run_global_timeout():
    try:
        result = await asyncio.wait_for(
            global_workflow(),
            timeout=GLOBAL_TIMEOUT_SECONDS,
        )

        return {
            "success": True,
            "result": result,
        }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "timeout": True,
            "error": "Global workflow timeout exceeded",
        }


# ============================================================
# TASK 16 VALIDATION
# ============================================================

async def main():

    print("TASK 16 - TIMEOUTS AND RETRIES")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"Per-node timeout: {NODE_TIMEOUT_SECONDS}s")
    print(f"Global timeout: {GLOBAL_TIMEOUT_SECONDS}s")
    print(f"Maximum attempts: {MAX_ATTEMPTS}")
    print(f"Initial retry interval: {INITIAL_INTERVAL}s")
    print(f"Maximum retry interval: {MAX_INTERVAL}s")
    print(f"Jitter: {JITTER}s")

    # --------------------------------------------------------
    # 1. TRANSIENT FAILURE + RETRY
    # --------------------------------------------------------

    print("\n1. RETRY TEST")
    print("-" * 70)

    operation = TransientOperation(
        failures_before_success=2
    )

    start = time.perf_counter()

    retry_result = await retry_with_backoff(
        operation
    )

    elapsed = time.perf_counter() - start

    print(f"Attempts: {retry_result['attempts']}")
    print(f"Retry delays: {retry_result['delays']}")
    print(f"Result: {retry_result['result']}")
    print(f"Elapsed: {elapsed:.3f}s")

    assert retry_result["success"] is True
    assert retry_result["attempts"] == 3
    assert len(retry_result["delays"]) == 2

    print("Retry test: PASSED")

    # --------------------------------------------------------
    # 2. PER-NODE TIMEOUT
    # --------------------------------------------------------

    print("\n2. PER-NODE TIMEOUT TEST")
    print("-" * 70)

    node_result = await run_node_with_timeout()

    print(node_result)

    assert node_result["success"] is False
    assert node_result["timeout"] is True

    print("Per-node timeout test: PASSED")

    # --------------------------------------------------------
    # 3. GLOBAL TIMEOUT
    # --------------------------------------------------------

    print("\n3. GLOBAL TIMEOUT TEST")
    print("-" * 70)

    global_result = await run_global_timeout()

    print(global_result)

    assert global_result["success"] is False
    assert global_result["timeout"] is True

    print("Global timeout test: PASSED")

    # --------------------------------------------------------
    # 4. FINAL VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TASK 16 VALIDATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())