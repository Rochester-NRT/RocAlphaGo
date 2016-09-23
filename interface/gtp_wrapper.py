import sys
import multiprocessing
import gtp
from AlphaGo import go
from AlphaGo.util import save_gamestate_to_sgf


def run_gnugo(sgf_file_name, command):
    from distutils import spawn
    if spawn.find_executable('gnugo'):
        from subprocess import Popen, PIPE
        p = Popen(['gnugo', '--chinese-rules', '--mode', 'gtp', '-l', sgf_file_name],
                  stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out_bytes = p.communicate(input=command)[0]
        return out_bytes.decode('utf-8')[2:]
    else:
        return ''


class ExtendedGtpEngine(gtp.Engine):

    recommended_handicaps = {
        2: "D4 Q16",
        3: "D4 Q16 D16",
        4: "D4 Q16 D16 Q4",
        5: "D4 Q16 D16 Q4 K10",
        6: "D4 Q16 D16 Q4 D10 Q10",
        7: "D4 Q16 D16 Q4 D10 Q10 K10",
        8: "D4 Q16 D16 Q4 D10 Q10 K4 K16",
        9: "D4 Q16 D16 Q4 D10 Q10 K4 K16 K10"
    }

    def call_gnugo(self, sgf_file_name, command):
        try:
            pool = multiprocessing.Pool(processes=1)
            result = pool.apply_async(run_gnugo, (sgf_file_name, command))
            output = result.get(timeout=10)
            pool.close()
            return output
        except multiprocessing.TimeoutError:
            pool.terminate()
            # if can't get answer from GnuGo, return no result
            return ''

    def cmd_time_left(self, arguments):
        pass

    def cmd_place_free_handicap(self, arguments):
        try:
            number_of_stones = int(arguments)
        except Exception:
            raise ValueError('Number of handicaps could not be parsed: {}'.format(arguments))
        if number_of_stones < 2 or number_of_stones > 9:
            raise ValueError('Invalid number of handicap stones: {}'.format(number_of_stones))
        vertex_string = ExtendedGtpEngine.recommended_handicaps[number_of_stones]
        self.cmd_set_free_handicap(vertex_string)
        return vertex_string

    def cmd_set_free_handicap(self, arguments):
        vertices = arguments.strip().split()
        moves = [gtp.parse_vertex(vertex) for vertex in vertices]
        self._game.place_handicaps(moves)

    def cmd_final_score(self, arguments):
        sgf_file_name = self._game.get_current_state_as_sgf()
        return self.call_gnugo(sgf_file_name, 'final_score\n')

    def cmd_final_status_list(self, arguments):
        sgf_file_name = self._game.get_current_state_as_sgf()
        return self.call_gnugo(sgf_file_name, 'final_status_list {}\n'.format(arguments))

    def cmd_load_sgf(self, arguments):
        pass

    def cmd_save_sgf(self, arguments):
        pass

    # def cmd_kgs_genmove_cleanup(self, arguments):
    #     return self.cmd_genmove(arguments)


class GTPGameConnector(object):
    """A class implementing the functions of a 'game' object required by the GTP
    Engine by wrapping a GameState and Player instance
    """

    def __init__(self, player):
        self._state = go.GameState(enforce_superko=True)
        self._player = player

    def clear(self):
        self._state = go.GameState(self._state.size, enforce_superko=True)

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
        self._state = go.GameState(n, enforce_superko=True)

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

    def get_current_state_as_sgf(self):
        from tempfile import NamedTemporaryFile
        temp_file = NamedTemporaryFile(delete=False)
        save_gamestate_to_sgf(self._state, '', temp_file.name)
        return temp_file.name

    def place_handicaps(self, vertices):
        actions = []
        for vertex in vertices:
            (x, y) = vertex
            actions.append((x - 1, y - 1))
        self._state.place_handicaps(actions)


def run_gtp(player_obj, inpt_fn=None, name="Gtp Player", version="0.0"):
    gtp_game = GTPGameConnector(player_obj)
    gtp_engine = ExtendedGtpEngine(gtp_game, name, version)
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
