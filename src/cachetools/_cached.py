"""Function decorator helpers."""

__all__ = ()

import functools
import threading
import warnings

# At least for now, the implementation prefers clarity and performance
# over ease of maintenance, thus providing separate wrappers for
# all valid combinations of decorator parameters lock, condition and
# info.


class _Pending:
    """Record of an in-flight function call for a single cache key.

    Used by the locked wrappers to make concurrent calls for the same
    key compute the value only once: the first thread to arrive stores
    a record in the pending table and calls the wrapped function, while
    subsequent threads wait on the record's event for its result.
    """

    __slots__ = ("event", "result", "exc")

    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.exc = None


def _condition_info(func, cache, key, lock, cond, info):
    hits = misses = 0
    pending = set()

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        k = key(*args, **kwargs)
        with lock:
            cond.wait_for(lambda: k not in pending)
            try:
                result = cache[k]
                hits += 1
                return result
            except KeyError:
                pending.add(k)
                misses += 1
        try:
            v = func(*args, **kwargs)
            with lock:
                try:
                    cache[k] = v
                except ValueError:
                    pass  # value too large
                return v
        finally:
            with lock:
                pending.remove(k)
                cond.notify_all()

    def cache_clear():
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = misses = 0

    def cache_info():
        with lock:
            return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _locked_info(func, cache, key, lock, info):
    hits = misses = 0
    pending = {}
    pending_lock = threading.Lock()

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        k = key(*args, **kwargs)
        with lock:
            try:
                result = cache[k]
                hits += 1
                return result
            except KeyError:
                pass  # key not found
        # The caller-provided lock is never held while calling func()
        # or waiting for a pending call, so distinct keys are computed
        # concurrently and non-reentrant locks are safe to use.
        with pending_lock:
            record = pending.get(k)
            if record is None:
                record = pending[k] = _Pending()
                misses += 1
                leader = True
            else:
                leader = False
        if not leader:
            # Another thread is already computing this key; wait for
            # its result and count it as a hit, so that misses always
            # equals the number of actual func() calls.
            record.event.wait()
            if record.exc is not None:
                raise record.exc
            with pending_lock:
                hits += 1
            return record.result
        try:
            v = func(*args, **kwargs)
            with lock:
                try:
                    # In case of a race condition, i.e. if another thread
                    # stored a value for this key while we were calling
                    # func(), prefer the cached value.
                    v = cache.setdefault(k, v)
                except ValueError:
                    pass  # value too large
            record.result = v
            return v
        except BaseException as e:
            # Waiters must receive the exact same exception.
            record.exc = e
            raise
        finally:
            # Remove the record before notifying waiters, so the
            # pending table is cleaned up even if this thread gets
            # interrupted while notifying.
            with pending_lock:
                pending.pop(k, None)
                record.event.set()

    def cache_clear():
        nonlocal hits, misses
        with lock:
            cache.clear()
            hits = misses = 0

    def cache_info():
        with lock:
            return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _unlocked_info(func, cache, key, info):
    hits = misses = 0

    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        k = key(*args, **kwargs)
        try:
            result = cache[k]
            hits += 1
            return result
        except KeyError:
            misses += 1
        v = func(*args, **kwargs)
        try:
            cache[k] = v
        except ValueError:
            pass  # value too large
        return v

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = misses = 0

    def cache_info():
        return info(hits, misses)

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper


def _uncached_info(func, info):
    misses = 0

    def wrapper(*args, **kwargs):
        nonlocal misses
        misses += 1
        return func(*args, **kwargs)

    def cache_clear():
        nonlocal misses
        misses = 0

    wrapper.cache_clear = cache_clear
    wrapper.cache_info = lambda: info(0, misses)
    return wrapper


def _condition(func, cache, key, lock, cond):
    pending = set()

    def wrapper(*args, **kwargs):
        k = key(*args, **kwargs)
        with lock:
            cond.wait_for(lambda: k not in pending)
            try:
                result = cache[k]
                return result
            except KeyError:
                pending.add(k)
        try:
            v = func(*args, **kwargs)
            with lock:
                try:
                    cache[k] = v
                except ValueError:
                    pass  # value too large
                return v
        finally:
            with lock:
                pending.remove(k)
                cond.notify_all()

    def cache_clear():
        with lock:
            cache.clear()

    wrapper.cache_clear = cache_clear
    return wrapper


def _locked(func, cache, key, lock):
    pending = {}
    pending_lock = threading.Lock()

    def wrapper(*args, **kwargs):
        k = key(*args, **kwargs)
        with lock:
            try:
                return cache[k]
            except KeyError:
                pass  # key not found
        # The caller-provided lock is never held while calling func()
        # or waiting for a pending call, so distinct keys are computed
        # concurrently and non-reentrant locks are safe to use.
        with pending_lock:
            record = pending.get(k)
            if record is None:
                record = pending[k] = _Pending()
                leader = True
            else:
                leader = False
        if not leader:
            # Another thread is already computing this key; wait for
            # its result.
            record.event.wait()
            if record.exc is not None:
                raise record.exc
            return record.result
        try:
            v = func(*args, **kwargs)
            with lock:
                try:
                    # In case of a race condition, i.e. if another thread
                    # stored a value for this key while we were calling
                    # func(), prefer the cached value.
                    v = cache.setdefault(k, v)
                except ValueError:
                    pass  # value too large
            record.result = v
            return v
        except BaseException as e:
            # Waiters must receive the exact same exception.
            record.exc = e
            raise
        finally:
            # Remove the record before notifying waiters, so the
            # pending table is cleaned up even if this thread gets
            # interrupted while notifying.
            with pending_lock:
                pending.pop(k, None)
                record.event.set()

    def cache_clear():
        with lock:
            cache.clear()

    wrapper.cache_clear = cache_clear
    return wrapper


def _unlocked(func, cache, key):
    def wrapper(*args, **kwargs):
        k = key(*args, **kwargs)
        try:
            return cache[k]
        except KeyError:
            pass  # key not found
        v = func(*args, **kwargs)
        try:
            cache[k] = v
        except ValueError:
            pass  # value too large
        return v

    wrapper.cache_clear = lambda: cache.clear()
    return wrapper


def _uncached(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.cache_clear = lambda: None
    return wrapper


def _wrapper(func, cache, key, lock=None, cond=None, info=None):
    if cache is None:
        warnings.warn(
            "@cachetools.cached(cache=None) is deprecated",
            DeprecationWarning,
            stacklevel=3,
        )

    if info is not None:
        if cache is None:
            wrapper = _uncached_info(func, info)
        elif cond is not None and lock is not None:
            wrapper = _condition_info(func, cache, key, lock, cond, info)
        elif cond is not None:
            wrapper = _condition_info(func, cache, key, cond, cond, info)
        elif lock is not None:
            wrapper = _locked_info(func, cache, key, lock, info)
        else:
            wrapper = _unlocked_info(func, cache, key, info)
    else:
        if cache is None:
            wrapper = _uncached(func)
        elif cond is not None and lock is not None:
            wrapper = _condition(func, cache, key, lock, cond)
        elif cond is not None:
            wrapper = _condition(func, cache, key, cond, cond)
        elif lock is not None:
            wrapper = _locked(func, cache, key, lock)
        else:
            wrapper = _unlocked(func, cache, key)
        wrapper.cache_info = None

    wrapper.cache = cache
    wrapper.cache_key = key
    wrapper.cache_lock = lock if lock is not None else cond
    wrapper.cache_condition = cond

    return functools.update_wrapper(wrapper, func)
