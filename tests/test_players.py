from AlphaGo.ai import ProbabilisticPolicyPlayer
import numpy as np
import unittest


class TestProbabilisticPolicyPlayer(unittest.TestCase):

    def test_temperature_increases_entropy(self):
        # helper function to get the entropy of a distribution
        def entropy(distribution):
            distribution = np.array(distribution).flatten()
            return -np.dot(np.log(distribution), distribution.T)
        player_low = ProbabilisticPolicyPlayer(None, temperature=0.9)
        player_high = ProbabilisticPolicyPlayer(None, temperature=1.1)

        distribution = np.random.random(361)
        distribution = distribution / distribution.sum()

        base_entropy = entropy(distribution)
        high_entropy = entropy(player_high.apply_temperature(distribution))
        low_entropy = entropy(player_low.apply_temperature(distribution))

        self.assertGreater(high_entropy, base_entropy)
        self.assertLess(low_entropy, base_entropy)

    def test_extreme_temperature_is_numerically_stable(self):
        player_low = ProbabilisticPolicyPlayer(None, temperature=1e-12)
        player_high = ProbabilisticPolicyPlayer(None, temperature=1e+12)

        distribution = np.random.random(361)
        distribution = distribution / distribution.sum()

        self.assertFalse(any(np.isnan(player_low.apply_temperature(distribution))))
        self.assertFalse(any(np.isnan(player_high.apply_temperature(distribution))))


if __name__ == '__main__':
    unittest.main()
