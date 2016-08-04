import numpy as np
from AlphaGo.models.policy import CNNPolicy
from interface.gtp_wrapper import run_gtp
from AlphaGo.ai import GreedyPolicyPlayer
#from AlphaGo.ai import MCTSPlayer
from AlphaGo.go import GameState
#from AlphaGo.util import save_gamestate_to_sgf


def policy_network(state):
	moves = state.get_legal_moves(include_eyes=False)
	# 'random' distribution over positions that is smallest
	# at (0,0) and largest at (18,18)
	probs = np.arange(361, dtype=np.float)
	probs = probs / probs.sum()
	return zip(moves, probs)

def value_network(state):
	# it's not very confident
	return 0.0

def rollout_policy(state):
	# just another policy network
	return policy_network(state)


MODEL = '/alphago/SLv1/my_model.json'
WEIGHTS = '/alphago/SLv1/weights.00002.hdf5'
policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)
policy_function = policy.eval_state

player = GreedyPolicyPlayer(policy)
#player = MCTSPlayer(policy_function, value_network, rollout_policy, lmbda=.5, c_puct=5, rollout_limit=500, playout_depth=5, n_search=3)

# Run gtp endpoint
#run_gtp(player)

# Run game
player = GreedyPolicyPlayer(policy)
gamestate = GameState(size=19)
counter = 0
# Play 10 games
while counter < 10:
	counter = counter + 1
	move = player.get_move(gamestate)
	gamestate.do_move(move)
	if gamestate.is_end_of_game:
		break

#save_gamestate_to_sgf(gamestate, "/alphago/heatmaps", "record.sgf", 'player', 'opponent')

