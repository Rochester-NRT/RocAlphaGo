import threading


class DLLNode(object):
    '''A single node in a doubly-linked list.
    '''

    __slots__ = ['key', 'value', 'next', 'prev']

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = self
        self.prev = self

    def insert_after(self, other):
        self.next.prev = other
        other.next = self.next
        self.next = other
        other.prev = self

    def insert_before(self, other):
        self.prev.next = other
        other.prev = self.prev
        self.prev = other
        other.next = self

    def splice(self):
        self.prev.next = self.next
        self.next.prev = self.prev
        self.next = self
        self.prev = self


class LRUCache(object):
    '''A least-recently used cache. This is a key-value store with a limit on the number of
       elements that can be stored. When elements are added that would make the size of the cache
       larger than its limit, the least-recently used key/value pair is deleted.

       Both writing (using put) and reading (using get) count as 'using' a key/value pair.

       The implementation uses a doubly-linked list to keep track of the order in which elements
       were accessed. self.sentinel.next is always the most-recently-used, and self.sentinel.prev is
       always the least-recently used. self.lookup is a dictionary mapping from keys to DLL nodes.
       Between hash table lookups and DLL operations, all updates are done in constant-time
       (amortized).

       Public functions are thread-safe (i.e. get() and put()), but private functions are not (i.e.
       _refresh() and _delete())
    '''

    __slots__ = ['sentinel', 'lookup', 'max_size', 'lock']

    def __init__(self, max_size=1e5):
        self.sentinel = DLLNode(None, None)
        self.lookup = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key):
        '''If 'key' is in the cache, return a tuple (True, value). Otherwise, return (False, None).
        '''
        with self.lock:
            if key in self.lookup:
                self._refresh(key)
                return True, self.lookup[key].value
            else:
                return False, None

    def _refresh(self, key):
        '''Move keyed item to the front of the DLL. key must be in the cache.
        '''
        node = self.lookup[key]
        node.splice()
        self.sentinel.insert_after(node)

    def put(self, key, value):
        '''Put value into cache or refresh existing value. If the cache size exceeds max_size, the
        least recently used key/value pair is removed.
        '''
        with self.lock:
            if key in self.lookup:
                self._refresh(key)
                self.lookup[key].value = value
            else:
                # Add this key/value pair to the cache.
                node = DLLNode(key, value)
                self.sentinel.insert_after(node)
                self.lookup[key] = node
                # Keep size below max_size.
                if len(self.lookup) > self.max_size:
                    # Delete least recently used key/value pair.
                    lru_key = self.sentinel.prev.key
                    self._delete(lru_key)

    def _delete(self, key):
        '''Remove an item from the cache. No error is raised if key is not in the cache.
        '''
        if key in self.lookup:
            self.lookup[key].splice()
            del self.lookup[key]


def lru_cache(max_size, key_fn=None):
    '''Decorator that wraps a function in an LRU cache. If no key_fn is given, the function's args
    are used as the cache key (they must be hashable).
    '''

    cache = LRUCache(max_size=max_size)

    def lru_decorator(fn):
        def lookup_or_execute(*args, **kwargs):
            key = tuple(args) if key_fn is None else key_fn(*args)
            is_cached, value = cache.get(key)
            if not is_cached:
                value = fn(*args, **kwargs)
                cache.put(key, value)
            return value
        return lookup_or_execute
    return lru_decorator
