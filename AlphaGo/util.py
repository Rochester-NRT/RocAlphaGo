<<<<<<< HEAD
﻿import sgf
import string
import six.moves.cPickle as pickle
from AlphaGo import go
import datetime
=======
import sgf
from AlphaGo import go

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

>>>>>>> latest

def flatten_idx(position, size):
	(x, y) = position
	return x * size + y


def unflatten_idx(idx, size):
	x, y = divmod(idx, size)
	return (x, y)


def _parse_sgf_move(node_value):
	"""Given a well-formed move string, return either PASS_MOVE or the (x, y) position
	"""
	if node_value == '' or node_value == 'tt':
		return go.PASS_MOVE
	else:
		row = LETTERS.index(node_value[1].upper())
		col = LETTERS.index(node_value[0].upper())
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
	for (gs, move, player) in sgf_iter_states(sgf_string, True):
		pass
	# gs has been updated in-place to the final state by the time
	# sgf_iter_states returns
	return gs


def sgf_iter_states(sgf_string, include_end=True):
	"""Iterates over (GameState, move, player) tuples in the first game of the given SGF file.

	Ignores variations - only the main line is returned.
	The state object is modified in-place, so don't try to, for example, keep track of it through time

	If include_end is False, the final tuple yielded is the penultimate state, but the state
	will still be left in the final position at the end of iteration because 'gs' is modified
	in-place the state. See sgf_to_gamestate
	"""
	collection = sgf.parse(sgf_string)
	game = collection[0]
	gs = _sgf_init_gamestate(game.root)
<<<<<<< HEAD
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
		
		
def gamestate_dump(gamestate, path="d:\tmp\gamestate.gs"):
	""" Dump game state to file
	"""
	output = open(path, 'wb')
	pickle.dump(gamestate, output)

def gamestate_to_sgf(gamestate,path="d:\\tmp\\gamestate.sgf"):
    
    alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    sgf_string = ""

    sgf_string += ";FF[%d]GM[%d]SZ[%d]CA[UTF-8]SO[GeekGo.com]BC[ja]WC[ja]EV[%s]" % (4,1,19,'GeekGo')
    sgf_string += "PB[%s]BR[%s]" % ('Computer1','9p')
    sgf_string += "PW[%s]WR[%s]" % ('Computer2','9p')
    sgf_string += "KM[%f]DT[%s]RE[%s]" % (6.5, datetime.datetime.now().strftime('%Y-%m-%d'),'Unknown')
    count = 1
    for (x,y) in gamestate.history:
        if(count & 1):
            sgf_string += ";B[%(x)s%(y)s]" % {'x':alphabet[x],'y':alphabet[y]}
        else:
            sgf_string += ";W[%(x)s%(y)s]" % {'x':alphabet[x],'y':alphabet[y]}
        count += 1 

    sgf_string = "(" + sgf_string + ")"

    try:
        f = open(path,'w')
        f.write(sgf_string)
    except:
        print "Error\n"
    else:
        f.close()
    return sgf_string
=======
	if game.rest is not None:
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
	if include_end:
		yield (gs, None, None)
>>>>>>> latest
