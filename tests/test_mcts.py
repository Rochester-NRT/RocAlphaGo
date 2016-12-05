from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS, TreeNode
from operator import itemgetter
import numpy as np
import unittest


class TestTreeNode(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()
        self.node = TreeNode(None, 1.0)

    def test_selection(self):
        self.node.expand(dummy_policy(self.gs))
        action, next_node = self.node.select()
        self.assertEqual(action, (18, 18))  # according to the dummy policy below
        self.assertIsNotNone(next_node)

    def test_expansion(self):
        self.assertEqual(0, len(self.node._children))
        self.node.expand(dummy_policy(self.gs))
        self.assertEqual(19 * 19, len(self.node._children))
        for a, p in dummy_policy(self.gs):
            self.assertEqual(p, self.node._children[a]._P)

    def test_update(self):
        self.node.expand(dummy_policy(self.gs))
        child = self.node._children[(18, 18)]
        # Note: the root must be updated first for the visit count to work.
        self.node.update(leaf_value=1.0, c_puct=5.0)
        child.update(leaf_value=1.0, c_puct=5.0)
        expected_score = 1.0 + 5.0 * dummy_distribution[-1] * 0.5
        self.assertEqual(expected_score, child.get_value())
        # After a second update, the Q value should be the average of the two, and the u value
        # should be multiplied by  sqrt(parent visits) / (node visits + 1) (which was simply equal
        # to 0.5 before)
        self.node.update(leaf_value=0.0, c_puct=5.0)
        child.update(leaf_value=0.0, c_puct=5.0)
        expected_score = 0.5 + 5.0 * dummy_distribution[-1] * np.sqrt(2.0) / 3.0
        self.assertEqual(expected_score, child.get_value())

    def test_update_recursive(self):
        # Assertions are identical to test_treenode_update.
        self.node.expand(dummy_policy(self.gs))
        child = self.node._children[(18, 18)]
        child.update_recursive(leaf_value=1.0, c_puct=5.0)
        expected_score = 1.0 + 5.0 * dummy_distribution[-1] / 2.0
        self.assertEqual(expected_score, child.get_value())
        child.update_recursive(leaf_value=0.0, c_puct=5.0)
        expected_score = 0.5 + 5.0 * dummy_distribution[-1] * np.sqrt(2.0) / 3.0
        self.assertEqual(expected_score, child.get_value())


class TestMCTS(unittest.TestCase):

    def setUp(self):
        self.gs = GameState()
        self.mcts = MCTS(dummy_value, dummy_policy, dummy_rollout, n_playout=2)

    def _count_expansions(self):
        """Helper function to count the number of expansions past the root using the dummy policy
        """
        node = self.mcts._root
        expansions = 0
        # Loop over actions in decreasing probability.
        for action, _ in sorted(dummy_policy(self.gs), key=itemgetter(1), reverse=True):
            if action in node._children:
                expansions += 1
                node = node._children[action]
            else:
                break
        return expansions

    def test_playout(self):
        self.mcts._playout(self.gs.copy(), 8)
        # Assert that the most likely child was visited (according to the dummy policy below).
        self.assertEqual(1, self.mcts._root._children[(18, 18)]._n_visits)
        # Assert that the search depth expanded nodes 8 times.
        self.assertEqual(8, self._count_expansions())

    def test_playout_with_pass(self):
        # Test that playout handles the end of the game (i.e. passing/no moves). Mock this by
        # creating a policy that returns nothing after 4 moves.
        def stop_early_policy(state):
            if len(state.history) <= 4:
                return dummy_policy(state)
            else:
                return []
        self.mcts = MCTS(dummy_value, stop_early_policy, stop_early_policy, n_playout=2)
        self.mcts._playout(self.gs.copy(), 8)
        # Assert that (18, 18) and (18, 17) are still only visited once.
        self.assertEqual(1, self.mcts._root._children[(18, 18)]._n_visits)
        # Assert that no expansions happened after reaching the "end" in 4 moves.
        self.assertEqual(5, self._count_expansions())

    def test_get_move(self):
        move = self.mcts.get_move(self.gs)
        self.mcts.update_with_move(move)
        # success if no errors

    def test_update_with_move(self):
        move = self.mcts.get_move(self.gs)
        self.gs.do_move(move)
        self.mcts.update_with_move(move)
        # Assert that the new root still has children.
        self.assertTrue(len(self.mcts._root._children) > 0)
        # Assert that the new root has no parent (the rest of the tree will be garbage collected).
        self.assertIsNone(self.mcts._root._parent)
        # Assert that the next best move according to the root is (18, 17), according to the
        # dummy policy below.
        self.assertEqual((18, 17), self.mcts._root.select()[0])


# A distribution over positions that is smallest at (0,0) and largest at (18,18)
dummy_distribution = np.arange(361, dtype=np.float)
dummy_distribution = dummy_distribution / dummy_distribution.sum()


def dummy_policy(state):
    moves = state.get_legal_moves(include_eyes=False)
    return zip(moves, dummy_distribution)


# Rollout is a clone of the policy function.
dummy_rollout = dummy_policy


def dummy_value(state):
    # it's not very confident
    return 0.0


if __name__ == '__main__':
    unittest.main()
