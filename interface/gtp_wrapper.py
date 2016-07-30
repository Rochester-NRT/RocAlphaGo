from AlphaGo import go
import gtp
import sys


class GTPGameConnector(object):
	"""A class implementing the functions of a 'game' object required by the GTP
	Engine by wrapping a GameState and Player instance
	"""

	def __init__(self, player):
		self._state = go.GameState()
		self._player = player

	def clear(self):
		self._state = go.GameState(self._state.size)

	def make_move(self, color, vertex):
		# vertex in GTP language is 1-indexed, whereas GameState's are zero-indexed
		try:
			if vertex == gtp.PASS:
				self._state.do_move(go.PASS_MOVE)
			else:
				(x, y) = vertex
				self._state.do_move((x - 1, y - 1), color)
			return True
		except go.IllegalMove:
			return False

	def set_size(self, n):
		self._state = go.GameState(n)

	def set_komi(self, k):
		self._state.komi = k

	def get_move(self, color):
		self._state.current_player = color
		move = self._player.get_move(self._state)
		if move == go.PASS_MOVE:
			return gtp.PASS
		else:
			(x, y) = move
			return (x + 1, y + 1)


def run_gtp(player_obj, inpt_fn=None):
	gtp_game = GTPGameConnector(player_obj)
	gtp_engine = gtp.Engine(gtp_game)
	if inpt_fn is None:
		inpt_fn = raw_input

	sys.stderr.write("GTP engine ready\n")
	sys.stderr.flush()
	while not gtp_engine.disconnect:
		inpt = inpt_fn()
		# handle either single lines at a time
		# or multiple commands separated by '\n'
		try:
			cmd_list = inpt.split("\n")
		except:
			cmd_list = [inpt]
		for cmd in cmd_list:
			engine_reply = gtp_engine.send(cmd)
			sys.stdout.write(engine_reply)
			sys.stdout.flush()
