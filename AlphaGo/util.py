import os
import itertools
import sgf
from AlphaGo import go

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


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
        # GameState expects (x, y) where x is column and y is row
        col = LETTERS.index(node_value[0].upper())
        row = LETTERS.index(node_value[1].upper())
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


def save_gamestate_to_sgf(gamestate, path, filename, black_player_name='Unknown',
                          white_player_name='Unknown', size=19, komi=7.5):
    """Creates a simplified sgf for viewing playouts or positions
    """
    str_list = []
    # Game info
    str_list.append('(;GM[1]FF[4]CA[UTF-8]')
    str_list.append('SZ[{}]'.format(size))
    str_list.append('KM[{}]'.format(komi))
    str_list.append('PB[{}]'.format(black_player_name))
    str_list.append('PW[{}]'.format(white_player_name))
    cycle_string = 'BW'
    # Handle handicaps
    if len(gamestate.handicaps) > 0:
        cycle_string = 'WB'
        str_list.append('HA[{}]'.format(len(gamestate.handicaps)))
        str_list.append(';AB')
        for handicap in gamestate.handicaps:
            str_list.append('[{}{}]'.format(LETTERS[handicap[0]].lower(),
                                            LETTERS[handicap[1]].lower()))
    # Move list
    for move, color in zip(gamestate.history, itertools.cycle(cycle_string)):
        # Move color prefix
        str_list.append(';{}'.format(color))
        # Move coordinates
        if move is None:
            str_list.append('[tt]')
        else:
            str_list.append('[{}{}]'.format(LETTERS[move[0]].lower(), LETTERS[move[1]].lower()))
    str_list.append(')')
    with open(os.path.join(path, filename), "w") as f:
        f.write(''.join(str_list))


def sgf_iter_states(sgf_string, include_end=True):
    """Iterates over (GameState, move, player) tuples in the first game of the given SGF file.

    Ignores variations - only the main line is returned.  The state object is
    modified in-place, so don't try to, for example, keep track of it through
    time

    If include_end is False, the final tuple yielded is the penultimate state,
    but the state will still be left in the final position at the end of
    iteration because 'gs' is modified in-place the state. See sgf_to_gamestate

    """
    collection = sgf.parse(sgf_string)
    game = collection[0]
    gs = _sgf_init_gamestate(game.root)
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
