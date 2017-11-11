import os
import itertools
import numpy as np
import sgf
from AlphaGo import go

# for board location indexing
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
       False for no.
       'resp' should be set to the default value assumed by the caller when
       user simply types ENTER.
       created by:
       http://code.activestate.com/recipes/541096-prompt-the-user-for-confirmation/
    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print 'please enter y or n.'
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False


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


def plot_network_output(scores, board, history, out_directory, output_file,
                        should_plot=False, western_column_notation=True):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError as e:
        print(
            'Failed to import matplotlib. This is an optional dependency of ' +
            'the RocAlphaGo project, so it is not included in the requirements file. ' +
            'You must install matplotlib yourself to use the plotting functions.')
        raise e

    from distutils.version import StrictVersion
    matplotlib_version = matplotlib.__version__
    if StrictVersion(matplotlib_version) < StrictVersion('1.5.1'):
        print('Your version of matplotlib might not support our use of it')

    # Initial matplotlib setup
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlim([0, board.size + 1])
    plt.ylim([0, board.size + 1])

    # Wooden background color
    ax.set_axis_bgcolor('#fec97b')
    plt.gca().invert_yaxis()

    # Setup ticks
    ax.tick_params(axis='both', length=0, width=0)
    # Western notation has the origin at the lower-left
    if western_column_notation:
        plt.xticks(range(1, board.size + 1), range(1, board.size + 1))
        plt.yticks(range(1, board.size + 1), reversed(range(1, board.size + 1)))
    # Traditional has the origin at the upper-left and uses letters minus 'I' along the top
    else:
        ax.xaxis.tick_top()
        plt.xticks(range(1, board.size + 1), [x for x in LETTERS[:board.size + 1] if x != 'I'])
        plt.yticks(range(1, board.size + 1), range(1, board.size + 1))

    # Draw grid
    for i in range(board.size):
        plt.plot([1, board.size], [i + 1, i + 1], lw=1, color='k', zorder=0)
    for i in range(board.size):
        plt.plot([i + 1, i + 1], [1, board.size], lw=1, color='k', zorder=0)

    # Display network heat plots
    reshaped = np.reshape(scores, (board.size, board.size))
    score_x_coords = []
    score_y_coords = []
    score_values = []
    for i in range(board.size):
        for j in range(board.size):
            if reshaped[i][j] * 100 >= 0.1:
                score_x_coords.append(i + 1)
                score_y_coords.append(j + 1)
                score_values.append(reshaped[i][j])
    min_seen = np.amin(scores)
    max_seen = np.amax(scores)
    norm = matplotlib.colors.Normalize(vmin=min_seen, vmax=max_seen)
    coloring = cm.ScalarMappable(norm=norm, cmap=cm.cool).to_rgba(score_values)
    plt.scatter(score_x_coords, score_y_coords, marker='o', s=700,
                c=coloring, edgecolor='k', zorder=1)

    # Display network scores on heat plots
    for i, txt in enumerate(score_values):
        ax.annotate('{0:.1f}'.format(txt * 100), (score_x_coords[i], score_y_coords[i]),
                    color='k', ha='center',
                    va='center', size=10, zorder=3)

    # Display stones already played
    stone_x_coords = []
    stone_y_coords = []
    stone_colors = []
    for i in range(board.size):
        for j in range(board.size):
            if board[i][j] != go.EMPTY:
                stone_x_coords.append(i + 1)
                stone_y_coords.append(j + 1)
                if board[i][j] == go.BLACK:
                    stone_colors.append(plt.to_rgb('black'))
                else:
                    stone_colors.append(plt.to_rgb('white'))
    plt.scatter(stone_x_coords, stone_y_coords, marker='o', edgecolors='k',
                s=700, c=stone_colors, zorder=4)

    # Place red marker on last move if it exists
    if len(history) != 0:
        # If last move was not pass
        if history[-1] != go.PASS_MOVE:
            last_move = history[-1]
            x_coord = last_move[0] + 1
            y_coord = last_move[1] + 1
            last_move = (x_coord, y_coord)
            plt.scatter(last_move[0], last_move[1], marker='s', color='r',
                        edgecolors='k', s=100, zorder=5)

    if output_file is not None:
        plt.savefig(os.path.join(out_directory, output_file), bbox_inches='tight')
    if should_plot:
        plt.show()
    plt.close()
