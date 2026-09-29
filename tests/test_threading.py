import threading
import time
import unittest
from typing import Any

from cachetools import LRUCache, cached, cachedmethod

count = 0


@cached(cache=LRUCache(1), condition=threading.Condition(), info=True)
def func():
    global count
    time.sleep(1.0)
    count += 1
    return count


@cached(cache=LRUCache(1), lock=threading.Lock(), info=True)
def locked_func():
    global count
    time.sleep(1.0)
    count += 1
    return 42


class ThreadingTest(unittest.TestCase):
    NTHREADS = 10

    TIMEOUT = 10

    cache: LRUCache[Any, int] = LRUCache(1)

    cond = threading.Condition()

    lock = threading.Lock()

    count = 0

    @cachedmethod(
        cache=lambda self: self.cache, condition=lambda self: self.cond, info=True
    )
    def method(self):
        time.sleep(1.0)
        self.count += 1
        return 42

    @cachedmethod(cache=lambda self: self.cache, lock=lambda self: self.lock, info=True)
    def locked_method(self):
        time.sleep(1.0)
        self.count += 1
        return 42

    def test_cached_stampede(self):
        global count
        count = 0
        threads = [threading.Thread(target=func) for i in range(self.NTHREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        self.assertEqual(count, 1)

        info = func.cache_info()
        self.assertEqual(info.hits, self.NTHREADS - 1)
        self.assertEqual(info.misses, 1)

    def test_cachedmethod_stampede(self):
        self.cache = LRUCache(1)
        self.count = 0
        threads = [threading.Thread(target=self.method) for i in range(self.NTHREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        self.assertEqual(self.count, 1)

        info = self.method.cache_info()
        self.assertEqual(info.hits, self.NTHREADS - 1)
        self.assertEqual(info.misses, 1)

    def test_cached_locked(self):
        global count
        count = 0
        threads = [threading.Thread(target=locked_func) for i in range(self.NTHREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        # Concurrent calls with identical keys are deduplicated: func is
        # called exactly once, and all threads return the same value (42).
        self.assertEqual(count, 1)

        info = locked_func.cache_info()
        self.assertEqual(info.hits + info.misses, self.NTHREADS)
        self.assertGreaterEqual(info.misses, 1)

    def test_cachedmethod_locked(self):
        self.cache = LRUCache(1)
        self.count = 0
        threads = [
            threading.Thread(target=self.locked_method) for i in range(self.NTHREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        # Multiple threads may compute the value, but setdefault ensures
        # the first cached result wins and all return 42.
        self.assertGreaterEqual(self.count, 1)

        info = self.locked_method.cache_info()
        self.assertEqual(info.hits + info.misses, self.NTHREADS)
        self.assertGreaterEqual(info.misses, 1)


class LockedSingleFlightTest(unittest.TestCase):
    NTHREADS = 5

    TIMEOUT = 10

    def test_same_key_computed_once(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        @cached(cache=LRUCache(1), lock=threading.Lock(), info=True)
        def func():
            calls.append(1)
            entered.set()
            release.wait(timeout=self.TIMEOUT)
            return object()

        results = []

        def call():
            results.append(func())

        threads = [threading.Thread(target=call) for _ in range(self.NTHREADS)]
        for t in threads:
            t.start()
        self.assertTrue(entered.wait(timeout=self.TIMEOUT))
        # give the remaining threads a chance to pile up on the pending call
        time.sleep(0.5)
        release.set()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), self.NTHREADS)
        # all callers receive the identical object
        self.assertTrue(all(r is results[0] for r in results))

        info = func.cache_info()
        self.assertEqual(info.hits + info.misses, self.NTHREADS)

    def test_different_keys_overlap(self):
        entered = {1: threading.Event(), 2: threading.Event()}
        errors = []

        @cached(cache=LRUCache(2), lock=threading.Lock())
        def func(x):
            entered[x].set()
            # wait for the other key's call to start: unless computations
            # with different keys overlap, this deadlocks and times out
            other = 2 if x == 1 else 1
            if not entered[other].wait(timeout=self.TIMEOUT):
                errors.append(RuntimeError("no overlap"))
            return x

        results = []

        def call(x):
            try:
                results.append(func(x))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call, args=(i,)) for i in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2 * self.TIMEOUT)
            self.assertFalse(t.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(sorted(results), [1, 2])

    def test_exception_shared_with_waiters(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        @cached(cache=LRUCache(1), lock=threading.Lock())
        def func():
            calls.append(1)
            entered.set()
            release.wait(timeout=self.TIMEOUT)
            raise ValueError("test error")

        errors = []

        def call():
            try:
                func()
            except ValueError as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(self.NTHREADS)]
        for t in threads:
            t.start()
        self.assertTrue(entered.wait(timeout=self.TIMEOUT))
        # give the remaining threads a chance to pile up on the pending call
        time.sleep(0.5)
        release.set()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        # the wrapped function raised only once...
        self.assertEqual(len(calls), 1)
        # ...but every caller received the very same exception
        self.assertEqual(len(errors), self.NTHREADS)
        self.assertTrue(all(e is errors[0] for e in errors))
        self.assertEqual(str(errors[0]), "test error")

        # the failed key must not be stuck: the next call recomputes
        with self.assertRaises(ValueError):
            func()
        self.assertEqual(len(calls), 2)
