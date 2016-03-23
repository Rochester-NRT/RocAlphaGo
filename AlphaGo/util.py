import sgf
import string
from AlphaGo import go

def _parse_sgf_move(node_value):
	"""Given a well-formed move string, return either PASS_MOVE or the (x, y) position
	"""
	if node_value == '' or node_value == 'tt':
		return go.PASS_MOVE
	else:
		row = string.letters.index(node_value[1])
		col = string.letters.index(node_value[0])
		# GameState expects (x, y) where x is column and y is row
		return (col, row)

def _sgf_init_gamestate(sgf_root):
	"""Helper function to set up a GameState object from the root node
	of an SGF file
	"""
	props = sgf_root.properties
	s_size = props.get('SZ', ['19'])[0]
	s_player = props.get('PL', ['B'])[0]
	# init board with specified size
	gs = go.GameState(int(s_size))
	# handle 'add black' property
	if 'AB' in props:
		for stone in props['AB']:
			gs.do_move(_parse_sgf_move(stone), go.BLACK)
	# handle 'add white' property
	if 'AW' in props:
		for stone in props['AW']:
			gs.do_move(_parse_sgf_move(stone), go.WHITE)
	# setup done; set player according to 'PL' property
	gs.current_player = go.BLACK if s_player == 'B' else go.WHITE
	return gs

def sgf_to_gamestate(sgf_string):
	"""Creates a GameState object from the first game in the given collection
	"""
	# Don't Repeat Yourself; parsing handled by sgf_iter_states
	for (gs, move, player) in sgf_iter_states(sgf_string):
		pass
	# gs has been updated in-place to the final state by the time
	# sgf_iter_states returns
	return gs

def sgf_iter_states(sgf_string):
	"""Iterates over (GameState, move, player) tuples in the first game of the given SGF file.

	Ignores variations - only the main line is returned.

	Note that the final tuple returned is the penultimate state,
	but because 'gs' is modified in-place the state is at the final
	position after iteration completes. See sgf_to_gamestate
	"""
	collection = sgf.parse(sgf_string)
	game = collection[0]
	gs = _sgf_init_gamestate(game.root)
	for node in game.rest:
		props = node.properties
		if 'W' in props:
			move = _parse_sgf_move(props['W'][0])
			player = go.WHITE
		elif 'B' in props:
			move = _parse_sgf_move(props['B'][0])
			player = go.BLACK
		yield (gs, move, player)
		# update state to n+1
		gs.do_move(move, player)
