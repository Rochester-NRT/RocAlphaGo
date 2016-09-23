from AlphaGo.go import GameState, BLACK, WHITE


def parse(boardstr):
    '''Parses a board into a gamestate, and returns the location of any moves
    marked with anything other than 'X', 'O', or '.'

    Rows are separated by '|', spaces are ignored.

    '''

    boardstr = boardstr.replace(' ', '')
    board_size = max(boardstr.index('|'), boardstr.count('|'))

    st = GameState(size=board_size)
    moves = {}

    for row, rowstr in enumerate(boardstr.split('|')):
        for col, c in enumerate(rowstr):
            if c == '.':
                continue  # ignore empty spaces
            elif c in 'BX#':
                st.do_move((row, col), color=BLACK)
            elif c in 'WO':
                st.do_move((row, col), color=WHITE)
            else:
                # move reference
                assert c not in moves, "{} already used as a move marker".format(c)
                moves[c] = (row, col)

    return st, moves
