"""Policy players"""
import numpy as np
from AlphaGo import go
from AlphaGo import mcts
from operator import itemgetter


class GreedyPolicyPlayer(object):
    """A player that uses a greedy policy (i.e. chooses the highest probability
       move each turn)
    """

    def __init__(self, policy_function, pass_when_offered=False, move_limit=None):
        self.policy = policy_function
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are sensible moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            max_prob = max(move_probs, key=itemgetter(1))
            return max_prob[0]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE


class ProbabilisticPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
       policy.

       By manipulating the 'temperature', moves can be pushed towards totally random
       (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, pass_when_offered=False, move_limit=None):
        assert(temperature > 0.0)
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.move_limit = move_limit

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def get_move(self, state):
        # check move limit
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE

        # check if pass was offered and we want to pass
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE

        # list with 'sensible' moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]

        # check if there are 'sensible' moves left to do
        if len(sensible_moves) > 0:
            move_probs = self.policy.eval_state(state, sensible_moves)
            # zip(*list) is like the 'transpose' of zip;
            # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
            moves, probabilities = zip(*move_probs)
            # apply 'temperature' to the distribution
            probabilities = self.apply_temperature(probabilities)
            # numpy interprets a list of tuples as 2D, so we must choose an
            # _index_ of moves then apply it in 2 steps
            choice_idx = np.random.choice(len(moves), p=probabilities)
            return moves[choice_idx]

        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [[move for move in st.get_legal_moves(include_eyes=False)]
                               for st in states]
        all_moves_distributions = self.policy.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            if len(move_probs) == 0 or len(states[i].history) > self.move_limit:
                move_list[i] = go.PASS_MOVE
            else:
                # this 'else' clause is identical to ProbabilisticPolicyPlayer.get_move
                moves, probabilities = zip(*move_probs)
                probabilities = np.array(probabilities)
                probabilities = probabilities ** self.beta
                probabilities = probabilities / probabilities.sum()
                choice_idx = np.random.choice(len(moves), p=probabilities)
                move_list[i] = moves[choice_idx]
        return move_list


class ProbabilisticGreedyPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
    policy.

    By manipulating the 'temperature', moves can be pushed towards totally random
    (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, policy_function, temperature=1.0, pass_when_offered=False,
                 move_limit=None, greedy_start=10):
        assert(temperature > 0.0)
        self.policy = policy_function
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.pass_when_offered = pass_when_offered
        self.greedy_start = greedy_start

    def apply_temperature(self, distribution):
        log_probabilities = np.log(distribution)
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # re-normalize the distribution
        return probabilities / probabilities.sum()

    def get_move(self, state):
        if self.move_limit is not None and len(state.history) > self.move_limit:
            return go.PASS_MOVE
        if self.pass_when_offered:
            if len(state.history) > 100 and state.history[-1] == go.PASS_MOVE:
                return go.PASS_MOVE
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        if len(sensible_moves) > 0:

            move_probs = self.policy.eval_state(state, sensible_moves)

            if len(state.history) > self.greedy_start:
                # greedy
                max_prob = max(move_probs, key=itemgetter(1))
                return max_prob[0]
            else:
                # probabilistic
                # zip(*list) is like the 'transpose' of zip;
                # zip(*zip([1,2,3], [4,5,6])) is [(1,2,3), (4,5,6)]
                moves, probabilities = zip(*move_probs)
                # apply 'temperature' to the distribution
                probabilities = self.apply_temperature(probabilities)
                # numpy interprets a list of tuples as 2D, so we must choose an
                # _index_ of moves then apply it in 2 steps
                choice_idx = np.random.choice(len(moves), p=probabilities)
                return moves[choice_idx]
        return go.PASS_MOVE


class MCTSPlayer(object):
    def __init__(self, value_function, policy_function, rollout_function, lmbda=.5, c_puct=5,
                 rollout_limit=500, playout_depth=40, n_playout=100):
        self.mcts = mcts.MCTS(value_function, policy_function, rollout_function, lmbda, c_puct,
                              rollout_limit, playout_depth, n_playout)

    def get_move(self, state):
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(state)
            self.mcts.update_with_move(move)
            return move
        # No 'sensible' moves available, so do pass move
        return go.PASS_MOVE


AI_LIST = {
    # "valuegreedy": GreedyValuePlayer,
    "policygreedy": GreedyPolicyPlayer,
    # "rolloutgreedy": GreedyRolloutPlayer,
    # "valueboth": ProbabilisticGreedyValuePlayer,
    "policyboth": ProbabilisticGreedyPolicyPlayer,
    # "rolloutboth": ProbabilisticGreedyRolloutPlayer,
    # "valueprobabilistic": ProbabilisticValuePlayer,
    "policyprobabilistic": ProbabilisticPolicyPlayer
    # "rolloutprobabilistic": ProbabilisticRolloutPlayer
}


def create_gtp(cmd_line_args=None):
    """Run gtp player.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Run gtp player.')
    # required args
    parser.add_argument('model_type', help="Model type eg. policy, value or rollout", choices=['policy', 'value', 'rollout'])  # noqa: E501
    parser.add_argument('player_type', help="Player type eg. greedy, probabilistic or both", choices=['greedy', 'probabilistic', 'both'])  # noqa: E501
    parser.add_argument('model', help="Path to a JSON model file.", default=None)  # noqa: E501
    parser.add_argument("weights", help="Name of .h5 weights file to load.", default=None)  # noqa: E501
    # optional args
    parser.add_argument("--temperature", help="Distribution temperature of player using probabilistic player-type (Default: 0.67)", type=float, default=0.67)  # noqa: E501
    parser.add_argument("--move-switch", help="Moves played with probabilistic before switching between player-type to greedy. (player-type 'both' only) (Default: 10)", type=int, default=10)  # noqa: E501
    parser.add_argument("--move-limit", help="Amount of moves before auto pass", type=int, default=None)  # noqa: E501
    parser.add_argument("--pass-when-offered", help="Turn on Accept_pass_when_offered mode.(only after move 100) (Default: False)", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # load model
    if args.model_type == "policy":

        from AlphaGo.models.policy import CNNPolicy
        model = CNNPolicy.load_model(args.model)
    elif args.model_type == "value":

        from AlphaGo.models.value import CNNValue
        model = CNNValue.load_model(args.model)
    elif args.model_type == "rollout":

        from AlphaGo.models.rollout import CNNRollout
        model = CNNRollout.load_model(args.model)

    # load weights
    model.model.load_weights(args.weights)

    # create player
    if args.player_type == "greedy":

        player = AI_LIST[args.model_type +
                         args.player_type](model, pass_when_offered=args.pass_when_offered,
                                           move_limit=args.move_limit)
    elif args.player_type == "probabilistic":

        player = AI_LIST[args.model_type +
                         args.player_type](model, temperature=args.temperature,
                                           pass_when_offered=args.pass_when_offered,
                                           move_limit=args.move_limit)
    elif args.player_type == "both":

        player = AI_LIST[args.model_type +
                         args.player_type](model, temperature=args.temperature,
                                           pass_when_offered=args.pass_when_offered,
                                           greedy_start=args.move_switch,
                                           move_limit=args.move_limit)

    # start gtp
    from interface.gtp_wrapper import run_gtp
    run_gtp(player)


if __name__ == '__main__':
    create_gtp()
