from AlphaGo.preprocessing.caching import LRUCache, lru_cache
import unittest


class TestLRUCache(unittest.TestCase):

    def setUp(self):
        self.cache = LRUCache(max_size=3)

    def testGetPutDelete(self):
        self.assertEqual(len(self.cache.lookup), 0)
        self.cache.put('a', 1)
        self.cache.put('b', 2)
        self.cache.put('c', 3)
        self.assertEqual(len(self.cache.lookup), 3)

        # 'c' should be cached and most-recently-used.
        # Current cache order: a b c
        is_cached, value = self.cache.get('c')
        self.assertTrue(is_cached)
        self.assertEqual(value, 3)

        # A 4th insertion should evict 'a', which is least-recently used.
        # Current cache order: a b c
        self.cache.put('d', 4)
        # Current cache order: b c d
        self.assertEqual(len(self.cache.lookup), 3)
        is_cached, value = self.cache.get('a')
        self.assertFalse(is_cached)
        self.assertIsNone(value)
        self.assertEqual(len(self.cache.lookup), 3)

        # 'b' is LRU. After querying it, 'c' should be LRU.
        # Current cache order: b c d
        self.cache.get('b')
        # Current cache order: c d b
        self.cache.put('e', 5)
        # Current cache order: d b e
        is_cached, value = self.cache.get('c')
        self.assertFalse(is_cached)
        self.assertEqual(len(self.cache.lookup), 3)

        # test deletion.
        self.cache._delete('d')
        # Current cache order: b e
        self.assertEqual(len(self.cache.lookup), 2)
        is_cached, value = self.cache.get('d')
        self.assertFalse(is_cached)


class TestDecorator(unittest.TestCase):

    def testNoKeyFn(self):
        @lru_cache(max_size=3)
        def fn(x, y=0):
            return x + y
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(2), 2)
        # fn should return the cached value here
        self.assertEqual(fn(1, y=100), 1)
        self.assertEqual(fn(2), 2)
        self.assertEqual(fn(3), 3)
        self.assertEqual(fn(4), 4)
        # key '1' is expired. Value will be recomputed.
        self.assertEqual(fn(1, y=100), 101)

    def testKeyFn(self):
        # Demonstrate key functions using absolute value of input as key.
        @lru_cache(max_size=3, key_fn=abs)
        def fn(x):
            return x + 1
        self.assertEqual(fn(1), 2)
        self.assertEqual(fn(2), 3)
        # fn should return the cached value here
        self.assertEqual(fn(-1), 2)
        self.assertEqual(fn(2), 3)
        self.assertEqual(fn(3), 4)
        self.assertEqual(fn(4), 5)
        # key '1' is expired. Value will be recomputed.
        self.assertEqual(fn(-1), 0)


if __name__ == '__main__':
    unittest.main()
