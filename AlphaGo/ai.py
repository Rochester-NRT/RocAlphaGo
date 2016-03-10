class GreedyPolicyPlayer(object):
    """A CNN player that uses a greedy policy (i.e. chooses the highest probability
       move at each point)
    """
    def __init__(self, policy_function):
        self.policy = policy_function

    def get_move(self, state):
        action_probs = self.policy.eval_state(state)
        max_prob = max(action_probs, key=lambda (a, p): p)
        return max_prob[0]
