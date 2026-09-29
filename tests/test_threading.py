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
        locked_func.cache_clear()
        threads = [threading.Thread(target=locked_func) for i in range(self.NTHREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        # With per-key single-flight, only one thread computes the value.
        self.assertEqual(count, 1)

        info = locked_func.cache_info()
        self.assertEqual(info.hits, self.NTHREADS - 1)
        self.assertEqual(info.misses, 1)

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
        cache = LRUCache(10)
        entered = threading.Event()
        release = threading.Event()
        calls = []

        @cached(cache=cache, lock=threading.Lock(), info=True)
        def func(n):
            calls.append(n)
            entered.set()
            release.wait(timeout=self.TIMEOUT)
            return object()

        results = []

        def run():
            results.append(func(1))

        threads = [threading.Thread(target=run) for i in range(self.NTHREADS)]
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
        # all callers receive the identical object
        self.assertTrue(all(r is results[0] for r in results))

        info = func.cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, self.NTHREADS - 1)

    def test_distinct_keys_computed_concurrently(self):
        cache = LRUCache(10)
        entered = {1: threading.Event(), 2: threading.Event()}
        errors = []

        @cached(cache=cache, lock=threading.Lock())
        def func(n):
            entered[n].set()
            # if distinct keys were serialized, this would time out
            if not entered[3 - n].wait(timeout=self.TIMEOUT):
                raise RuntimeError("distinct keys not computed concurrently")
            return n

        def run(n):
            try:
                func(n)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run, args=(n,)) for n in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.TIMEOUT + 1)
            self.assertFalse(t.is_alive())

        self.assertEqual(errors, [])

    def test_exception_propagates_to_waiters(self):
        cache = LRUCache(10)
        entered = threading.Event()
        release = threading.Event()
        fail = True

        @cached(cache=cache, lock=threading.Lock())
        def func(n):
            entered.set()
            release.wait(timeout=self.TIMEOUT)
            if fail:
                raise ValueError("boom")
            return n

        errors = []

        def run():
            try:
                func(1)
            except ValueError as e:
                errors.append(e)

        threads = [threading.Thread(target=run) for i in range(self.NTHREADS)]
        for t in threads:
            t.start()
        self.assertTrue(entered.wait(timeout=self.TIMEOUT))
        # give the remaining threads a chance to pile up on the pending call
        time.sleep(0.5)
        release.set()
        for t in threads:
            t.join(timeout=self.TIMEOUT)
            self.assertFalse(t.is_alive())

        # all callers receive the identical exception, unwrapped
        self.assertEqual(len(errors), self.NTHREADS)
        self.assertTrue(all(e is errors[0] for e in errors))
        self.assertEqual(str(errors[0]), "boom")
        self.assertEqual(len(cache), 0)

        # the pending entry must have been cleaned up, so the key is
        # not stuck and a later call computes the value again
        fail = False
        self.assertEqual(func(1), 1)
