from AlphaGo.models.policy import CNNPolicy
from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.util import plot_network_output


MODEL = '/alphago/SLv1/my_model.json'
WEIGHTS = '/alphago/SLv1/weights.00002.hdf5'
policy = CNNPolicy.load_model(MODEL)
policy.model.load_weights(WEIGHTS)
policy_function = policy.eval_state

player = GreedyPolicyPlayer(policy)

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

plot_network_output(gamestate, "", "record.sgf", 'player', 'opponent')
