"""
Tests for GEE Circuit Breaker implementation.

These tests verify the circuit breaker pattern works correctly to prevent
cascade failures when GEE is experiencing issues.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.data.processors import (
    GEECircuitBreaker,
    CircuitBreakerOpenError
)


def create_circuit_breaker(failure_threshold=5, reset_timeout=300.0, half_open_max_calls=3):
    """Factory function to create a fresh circuit breaker for testing."""
    # Reset singleton state
    GEECircuitBreaker._instance = None
    # Create new instance
    cb = GEECircuitBreaker.__new__(GEECircuitBreaker)
    cb._initialized = False
    cb.__init__(failure_threshold, reset_timeout, half_open_max_calls)
    return cb


class TestGEECircuitBreaker:
    """Tests for GEECircuitBreaker class."""

    def setup_method(self):
        """Reset circuit breaker singleton before each test."""
        # Reset singleton state
        GEECircuitBreaker._instance = None
        # Also reset the _initialized flag on any existing instance
        if hasattr(GEECircuitBreaker, '_instance') and GEECircuitBreaker._instance:
            GEECircuitBreaker._instance._initialized = False

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = create_circuit_breaker(failure_threshold=3)
        assert cb.state == GEECircuitBreaker.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_can_execute_when_closed(self):
        """Requests are allowed when circuit is closed."""
        cb = create_circuit_breaker(failure_threshold=3)
        assert cb.can_execute() is True
        assert cb.can_execute() is True
        assert cb.can_execute() is True

    def test_failure_count_increments(self):
        """Failure count increments on each failure."""
        cb = create_circuit_breaker(failure_threshold=5)

        cb.record_failure(Exception("Error 1"))
        assert cb.failure_count == 1

        cb.record_failure(Exception("Error 2"))
        assert cb.failure_count == 2

        cb.record_failure(Exception("Error 3"))
        assert cb.failure_count == 3

    def test_success_resets_failure_count(self):
        """Success resets the failure count."""
        cb = create_circuit_breaker(failure_threshold=5)

        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == GEECircuitBreaker.CLOSED

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        cb = create_circuit_breaker(failure_threshold=3)

        cb.record_failure(Exception("Error 1"))
        assert cb.state == GEECircuitBreaker.CLOSED

        cb.record_failure(Exception("Error 2"))
        assert cb.state == GEECircuitBreaker.CLOSED

        cb.record_failure(Exception("Error 3"))
        assert cb.state == GEECircuitBreaker.OPEN
        assert cb.can_execute() is False

    def test_open_circuit_blocks_requests(self):
        """Open circuit blocks new requests."""
        cb = create_circuit_breaker(failure_threshold=2, reset_timeout=60)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))

        assert cb.state == GEECircuitBreaker.OPEN
        assert cb.can_execute() is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after reset timeout."""
        cb = create_circuit_breaker(failure_threshold=2, reset_timeout=0.1)  # 100ms timeout

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.state == GEECircuitBreaker.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should transition to half-open on next can_execute() call
        assert cb.can_execute() is True
        assert cb.state == GEECircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        """Half-open state allows limited test requests."""
        cb = create_circuit_breaker(
            failure_threshold=2,
            reset_timeout=0.1,
            half_open_max_calls=2
        )

        # Open then transition to half-open
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        time.sleep(0.15)

        # First call transitions to half-open (doesn't count against limit)
        assert cb.can_execute() is True
        assert cb.state == GEECircuitBreaker.HALF_OPEN

        # Second call allowed (call 1 of 2)
        assert cb.can_execute() is True

        # Third call allowed (call 2 of 2)
        assert cb.can_execute() is True

        # Fourth call blocked (exceeded limit of 2)
        assert cb.can_execute() is False

    def test_half_open_closes_on_success(self):
        """Half-open circuit closes on successful request."""
        cb = create_circuit_breaker(failure_threshold=2, reset_timeout=0.1)

        # Open then transition to half-open
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        time.sleep(0.15)
        cb.can_execute()  # Transition to half-open

        # Success should close the circuit
        cb.record_success()
        assert cb.state == GEECircuitBreaker.CLOSED
        assert cb.failure_count == 0

    def test_half_open_reopens_on_failure(self):
        """Half-open circuit reopens on failed request."""
        cb = create_circuit_breaker(failure_threshold=2, reset_timeout=0.1)

        # Open then transition to half-open
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        time.sleep(0.15)
        cb.can_execute()  # Transition to half-open
        assert cb.state == GEECircuitBreaker.HALF_OPEN

        # Failure should reopen the circuit
        cb.record_failure(Exception("Error in half-open"))
        assert cb.state == GEECircuitBreaker.OPEN

    def test_get_status_returns_correct_info(self):
        """get_status() returns correct circuit state information."""
        cb = create_circuit_breaker(failure_threshold=3, reset_timeout=60)

        status = cb.get_status()
        assert status['state'] == GEECircuitBreaker.CLOSED
        assert status['failure_count'] == 0
        assert status['failure_threshold'] == 3
        assert status['reset_timeout'] == 60
        assert status['time_until_retry'] == 0

    def test_get_status_shows_retry_time_when_open(self):
        """get_status() shows time until retry when circuit is open."""
        cb = create_circuit_breaker(failure_threshold=2, reset_timeout=60)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))

        status = cb.get_status()
        assert status['state'] == GEECircuitBreaker.OPEN
        assert status['time_until_retry'] > 0
        assert status['time_until_retry'] <= 60

    def test_reset_returns_to_closed_state(self):
        """reset() returns circuit to initial closed state."""
        cb = create_circuit_breaker(failure_threshold=2)

        # Open the circuit
        cb.record_failure(Exception("Error 1"))
        cb.record_failure(Exception("Error 2"))
        assert cb.state == GEECircuitBreaker.OPEN

        # Reset should return to closed
        cb.reset()
        assert cb.state == GEECircuitBreaker.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_singleton_pattern(self):
        """Circuit breaker uses singleton pattern."""
        # First, reset singleton
        GEECircuitBreaker._instance = None

        cb1 = GEECircuitBreaker()
        cb2 = GEECircuitBreaker()

        assert cb1 is cb2

        # State is shared
        cb1.record_failure(Exception("Error"))
        assert cb2.failure_count == 1

    def test_thread_safety(self):
        """Circuit breaker is thread-safe."""
        import threading

        cb = create_circuit_breaker(failure_threshold=100)

        def record_failures():
            for _ in range(10):
                cb.record_failure(Exception("Error"))

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All failures should be recorded
        assert cb.failure_count == 100


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_error_message(self):
        """Error contains informative message."""
        error = CircuitBreakerOpenError("Circuit is open")
        assert "Circuit is open" in str(error)

    def test_is_exception_subclass(self):
        """Error is a proper Exception subclass."""
        error = CircuitBreakerOpenError("Test")
        assert isinstance(error, Exception)


class TestGEETimeout:
    """Tests for gee_with_timeout function and GEETimeoutError."""

    def test_successful_execution(self):
        """Function that completes within timeout returns normally."""
        from src.data.processors import gee_with_timeout

        def fast_func():
            return "success"

        result = gee_with_timeout(fast_func, timeout=5, operation_name="fast test")
        assert result == "success"

    def test_timeout_raises_error(self):
        """Function that exceeds timeout raises GEETimeoutError."""
        from src.data.processors import gee_with_timeout, GEETimeoutError

        def slow_func():
            time.sleep(10)
            return "should not reach"

        with pytest.raises(GEETimeoutError) as exc_info:
            gee_with_timeout(slow_func, timeout=0.5, operation_name="slow test")

        assert "slow test" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    def test_exception_propagation(self):
        """Exceptions from the wrapped function are propagated."""
        from src.data.processors import gee_with_timeout

        def error_func():
            raise ValueError("test error")

        with pytest.raises(ValueError) as exc_info:
            gee_with_timeout(error_func, timeout=5, operation_name="error test")

        assert "test error" in str(exc_info.value)

    def test_return_value_types(self):
        """Various return types are handled correctly."""
        from src.data.processors import gee_with_timeout

        # Test dict return
        assert gee_with_timeout(lambda: {"key": "value"}, timeout=5) == {"key": "value"}

        # Test list return
        assert gee_with_timeout(lambda: [1, 2, 3], timeout=5) == [1, 2, 3]

        # Test None return
        assert gee_with_timeout(lambda: None, timeout=5) is None

        # Test numeric return
        assert gee_with_timeout(lambda: 42.5, timeout=5) == 42.5

    def test_thread_safety(self):
        """Timeout wrapper is thread-safe for parallel use."""
        from src.data.processors import gee_with_timeout
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = gee_with_timeout(
                    lambda: f"worker_{worker_id}",
                    timeout=5,
                    operation_name=f"worker {worker_id}"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r.startswith("worker_") for r in results)

    def test_default_timeout(self):
        """Default timeout constant is reasonable."""
        from src.data.processors import GEE_API_TIMEOUT

        # Should be at least 60 seconds for GEE operations
        assert GEE_API_TIMEOUT >= 60
        # But not excessively long
        assert GEE_API_TIMEOUT <= 300


class TestGEETimeoutError:
    """Tests for GEETimeoutError exception."""

    def test_error_message(self):
        """Error contains informative message."""
        from src.data.processors import GEETimeoutError

        error = GEETimeoutError("Operation timed out after 120 seconds")
        assert "timed out" in str(error)
        assert "120" in str(error)

    def test_is_exception_subclass(self):
        """Error is a proper Exception subclass."""
        from src.data.processors import GEETimeoutError

        error = GEETimeoutError("Test")
        assert isinstance(error, Exception)
