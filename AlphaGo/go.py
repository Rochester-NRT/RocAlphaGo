import numpy as np

WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None

class GameState(object):
    """State of a game of Go and some basic functions to interact with it
    """

    def __init__(self, size=19):
        self.board = np.zeros((size, size))
        self.board.fill(EMPTY)
        self.size = size
        self.turns_played = 0
        self.current_player = BLACK
        self.ko = None
        self.history = []
        self.num_black_prisoners = 0
        self.num_white_prisoners = 0

    def copy(self):
        """get a copy of this Game state
        """
        other = GameState(self.size)
        other.board = self.board.copy()
        other.turns_played = self.turns_played
        other.current_player = self.current_player
        other.ko = self.ko
        other.history = list(self.history)
        other.num_black_prisoners = self.num_black_prisoners
        other.num_white_prisoners = self.num_white_prisoners
        return other

    def liberty_count(self, position):
        """Count liberty of a single position (maxium = 4).

        Keyword arguments:
        position -- a tuple of (x, y)
        x being the column index of the position we want to calculate the liberty
        y being the row index of the position we want to calculate the liberty

        Return:
        q -- A interger in [0, 4]. The count of liberty of the input single
        position
        """
        return len(self.liberty_pos(position))

    def liberty_count_group(self, group):
        """Count liberty of a single position (maxium = 4).

        Keyword arguments:
        position -- a tuple of (x, y)
        x being the column index of the position we want to calculate the liberty
        y being the row index of the position we want to calculate the liberty

        Return:
        q -- The liberty count for entire group
        position
        """
        return len(self.liberty_pos_group(group))

    def liberty_pos(self, position):
        """Record the liberty positions of a single position.

        Keyword arguments:
        position -- a tuple of (x, y)
        x being the column index of the position we want to calculate the liberty
        y being the row index of the position we want to calculate the liberty

        Return:
        pos -- Return a list of tuples consist of (x, y)s which are the liberty
        positions on the input single position. len(pos) <= 4
        """
        (x, y) = position
        pos = []
        if x + 1 < self.size and self.board[x + 1][y] == EMPTY:
            pos.append((x + 1, y))
        if y + 1 < self.size and self.board[x][y + 1] == EMPTY:
            pos.append((x, y + 1))
        if x - 1 >= 0 and self.board[x - 1][y] == EMPTY:
            pos.append((x - 1, y))
        if y - 1 >= 0 and self.board[x][y - 1] == EMPTY:
            pos.append((x, y - 1))
        return pos

    def liberty_pos_group(self, group):
        """Get all liberty positions around group.

        Args:
        group (list of position tuples corresponding to connected group)

        Return:
        pos (list of position tuples corresponding to liberties of group)
        """
        pos = []
        for p in group:
            lib = self.liberty_pos(p)
            if len(lib) > 0:
                pos.extend(lib)
        pos = list(set(pos))  # Remove redundant tuples
        return pos

    def get_neighbor(self, position):
        """An auxiliary function for update_current_liberties. This function looks around
        locally in 4 directions. That is, we just pick one position and look to
        see if there are same-color neighbors around it.

        Keyword arguments:
        position -- a tuple of (x, y)
        x being the column index of the position in consideration
        y being the row index of the posisiton in consideration

        Return:
        neighbor -- Return a list of tuples consist of (x, y)s which are the
        same-color neighbors of the input single position.
        len(neighbor_set) <= 4
        """
        (x, y) = position
        neighbor_set = []
        if y+1 < self.size and self.board[x][y] == self.board[x][y+1]:
            neighbor_set.append((x, y+1))
        if x+1 < self.size and self.board[x][y] == self.board[x+1][y]:
            neighbor_set.append((x+1, y))
        if x-1 >= 0 and self.board[x][y] == self.board[x-1][y]:
            neighbor_set.append((x-1, y))
        if y-1 >= 0 and self.board[x][y] == self.board[x][y-1]:
            neighbor_set.append((x, y-1))
        return neighbor_set

    def get_group(self, position):
        """An auxiliary function for update_current_liberties. This function performs the
        visiting process to identify a connected group of the same color.

        Keyword arguments:
        position -- a tuple of (x, y)
        x being the column index of the starting position of the search
        y being the row index of the starting position of the search

        Return:
        neighbor_set -- Return a list of (x, y) tuples corresponding to a cluster of
        stones belonging to one player, which contains the input single position.
        len(neighbor_set) is size of the cluster, can be large.
        """
        (x, y) = position
        # handle case where there is no piece at (x,y)
        if self.board[x][y] == EMPTY:
            return set()
        # A list for record the places we visited in the process
        # default to the starting position to handle the case where there are no neighbors (group size is 1)
        visited = [(x, y)]
        # A list for the the places we still want to visit
        to_visit = self.get_neighbor((x, y))
        while len(to_visit) != 0:
            for n in to_visit:
                # append serve as the actual visit
                visited.append(n)
                # take off the places already visited from the wish list
                to_visit.remove(n)
            # With the cluster we have now, we look around even further
            for v in visited:
                # we try to look for same-color neighbors for each one which we already visited
                for n in self.get_neighbor(v):
                    # we don't need to consider the places we already visited when we're looking
                    if n not in visited:
                        to_visit.append(n)
        neighbor_list = list(set(visited))
        return neighbor_list

    def update_current_liberties(self):
        """Calculate the liberty values of the whole board

        Keyword arguments:
        None. We just need the board itself.

        Return:
        A matrix self.size * self.size, with entries of the liberty number of
        each position on the board.
        Empty spaces have liberty 0. Instead of the single stone liberty, we
        consider the liberty of the
        group/cluster of the same color the position is in.
        """

        curr_liberties = np.ones((self.size, self.size)) * (-1)

        for x in range(0, self.size):
            for y in range(0, self.size):

                if self.board[x][y] == EMPTY:
                    continue

                # get the members in the cluster and then calculate their liberty positions
                lib_set = set()
                neighbors = self.get_group((x, y))
                for n in neighbors:
                    lib_set |= set(self.liberty_pos(n))

                curr_liberties[x][y] = len(lib_set)
        return curr_liberties

    def is_suicide(self, action):
        """return true if having current_player play at <action> would be suicide
        """
        tmp = self.copy()
        tmp.board[action] = tmp.current_player
        zero_liberties = tmp.update_current_liberties() == 0
        other_player = tmp.board == -tmp.current_player
        to_remove = np.logical_and(zero_liberties, other_player)
        tmp.board[to_remove] = EMPTY
        return tmp.update_current_liberties()[action] == 0

    def is_legal(self, action):
        """determine if the given action (x,y tuple) is a legal move

        note: we only check ko, not superko at this point (TODO?)
        """
        # passing move
        if action is PASS_MOVE:
            return True
        (x, y) = action
        empty = self.board[x][y] == EMPTY
        on_board = x >= 0 and y >= 0 and x < self.size and y < self.size
        suicide = self.is_suicide(action)
        ko = action == self.ko
        return on_board and (not suicide) and (not ko) and empty

    def is_eye(self, position, owner):
        """returns whether the position is empty and is surrounded by all stones of 'owner'
        """
        (x, y) = position
        if self.board[x, y] != EMPTY:
            return False

        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for (nx, ny) in neighbors:
            if nx >= 0 and ny >= 0 and nx < self.size and ny < self.size:
                if self.board[nx, ny] != owner:
                    return False
        return True

    def get_legal_moves(self):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal((x, y)):
                    moves.append((x, y))
        return moves

    def is_ladder_capture(self, action):
        """A move is a 'ladder capture' if:
        1) after this move, opponent has only one liberty
        2) and after opponent plays on that liberty to escape, this creates
           exactly 2 liberties
        (Note that (1) doesn't have to be directly caused by the move we're
        considering. The atari may have been played on a previous turn.)
        """
        (x, y) = action
        # Play out from point of view of opponent getting captured by ladder
        tmp = self.copy()
        tmp.do_move(action)  # Do move=action
        # Check liberties of all pieces
        board_libs = tmp.update_current_liberties()
        # Find locations where opponent has only 1 liberty
        one_lib = board_libs == 1  # All board positions with one liberty
        if np.any(one_lib):
            print 'Found 1-libs on board...'
            # select only those 1-libs that belong to opponent
            one_lib_opp = np.logical_and(one_lib, tmp.board == tmp.current_player)
            # further select only those within the neighborhood of 'action'
            # neighbors = self.get_neighbor(action)
            # X_n, Y_n = neighbors
            # BP()
            # # for (x, y) in neighbors:
            # #     if 

            if np.any(one_lib_opp):
                print 'Found 1-libs belonging to opponent...'
                x_1lib, y_1lib = np.where(
                    np.logical_and(one_lib, tmp.board == tmp.current_player))
                # Convert from numpy to tuples
                tuples_one_lib_opp = [(x_, y_) for x_, y_ in zip(x_1lib, y_1lib)]
                print "Opponent 1-lib positions: ", tuples_one_lib_opp
                # For each opponent stone with 1 liberty, simulate if
                # move to its liberty would result in 2 liberties
                for t in tuples_one_lib_opp:
                    print "Testing tuple ", t
                    tmp1 = tmp.copy()
                    # Get open space
                    if tmp1.liberty_count(t) == 1:
                        libs = tmp1.liberty_pos(t)
                        assert len(libs) == 1  # If not, liberty_count was mistaken
                        lib = libs[0]
                        tmp1.do_move(lib)
                        # Now check liberties at t
                        n_libs = tmp1.liberty_count(lib)
                        if n_libs == 2:
                            # We've met our criteria. 'action' is ladder capture
                            return True
                        print "Didn't work. There were {} resultant liberties".format(n_libs)
                    else:
                        "Zero liberties. Trying next tuple..."
                # We've checked all opponent 1-libs, and none resulted in
                # exactly 2 libs, so 'action' is not a ladder capture.
                print "None of the opponent 1-lib plays resulted in 2 liberties. Not a ladder."
                return False
            else:
                print "Opponent had no 1-libs. Not a ladder."
                return False  # Not a ladder, because opponent has no 1-lib groups
        else:
            print "Found no 1-libs. Not a ladder."
            # Not a ladder, because no 1-lib groups on board
            return False

    def is_ladder_escape(self, action):
        """Version 1: A move is a 'ladder escape' if a) player is currently trapped
        in ladder and there is exactly one liberty for group of stones in the
        ladder, and b) playing at that liberty results in more than 2 liberties

        Version 2: A move is a 'ladder escape' if afterwards, opponent no longer has an
        option for ladder capture. Test by looking over all legal opponent
        moves to see if there are any ladder captures.
        """
        from ipdb import set_trace as BP
        # First, check if currently in a ladder by seeing if previous opponent
        # move was a ladder capture. (Sufficient check, because every
        # subsequent offensive by the capturer in a ladder sequence should
        # also be a ladder capture)
        if self.prev.is_ladder_capture(self.history[-1]):
            # Then, simulate move...
            tmp = self.copy()
            tmp.do_move(action)
            # ...and count liberties
            lib_count = tmp.liberty_count_group(tmp.get_group(action))
            BP()
            if lib_count > 2:
                return True

            # Version 2
            # # and test if opponent has any ladder capture options
            # for move in tmp.get_legal_moves():
            #     if tmp.is_ladder_capture(move):
            #         # Opponent still has a ladder capture available, so not a
            #         # ladder escape
            #         return False
            # return True
        return False

    def do_move(self, action):
        """Play current_player's color at (x,y)

        If it is a legal move, current_player switches to the other player
        If not, an IllegalMove exception is raised
        """
        # Hold onto previous state for use in is_ladder_escape
        self.prev = self.copy()
        if self.is_legal(action):
            # reset ko
            self.ko = None
            if action is not PASS_MOVE:
                (x, y) = action
                self.board[x][y] = self.current_player
                # check liberties for captures
                liberties = self.update_current_liberties()
                zero_liberties = liberties == 0
                other_player = self.board == -self.current_player
                captured_stones = np.logical_and(zero_liberties, other_player)
                capture_occurred = np.any(captured_stones)  # note EMPTY spaces are -1
                if capture_occurred:
                    # clear pieces
                    self.board[captured_stones] = EMPTY
                    # count prisoners
                    num_captured = np.sum(captured_stones)
                    if self.current_player == BLACK:
                        self.num_white_prisoners += num_captured
                    else:
                        self.num_black_prisoners += num_captured
                    if num_captured == 1:
                        xcoord, ycoord = np.where(captured_stones)
                        self.ko = (xcoord[0], ycoord[0])
            # next turn
            self.current_player = -self.current_player
            self.turns_played += 1
            self.history.append(action)
        else:
            raise IllegalMove(str(action))

    # def do_move_copy(self, gs, action):
    #     """Play current_player's color at (x,y) on copy of games state, gs

    #     If it is a legal move, current_player switches to the other player
    #     If not, an IllegalMove exception is raised
    #     """
    #     # Hold onto previous state for use in is_ladder_escape
    #     gs.prev = gs.copy()
    #     if gs.is_legal(action):
    #         # reset ko
    #         gs.ko = None
    #         if action is not PASS_MOVE:
    #             (x, y) = action
    #             gs.board[x][y] = gs.current_player
    #             # check liberties for captures
    #             liberties = gs.update_current_liberties()
    #             zero_liberties = liberties == 0
    #             other_player = gs.board == -gs.current_player
    #             captured_stones = np.logical_and(zero_liberties, other_player)
    #             capture_occurred = np.any(captured_stones)  # note EMPTY spaces are -1
    #             if capture_occurred:
    #                 # clear pieces
    #                 gs.board[captured_stones] = EMPTY
    #                 # count prisoners
    #                 num_captured = np.sum(captured_stones)
    #                 if gs.current_player == BLACK:
    #                     gs.num_white_prisoners += num_captured
    #                 else:
    #                     gs.num_black_prisoners += num_captured
    #                 if num_captured == 1:
    #                     xcoord, ycoord = np.where(captured_stones)
    #                     gs.ko = (xcoord[0], ycoord[0])
    #         # next turn
    #         gs.current_player = -gs.current_player
    #         gs.turns_played += 1
    #         gs.history.append(action)
    #     else:
    #         raise IllegalMove(str(action))
    #     return gs

    def symmetries(self):
        """returns a list of 8 GameState objects:
        all reflections and rotations of the current board

        does not check for duplicates
        """

        # we use numpy's built-in array symmetry routines for self.board.
        # but for all xy pairs (i.e. self.ko and self.history), we need to
        # know how to rotate a tuple (x,y) into (new_x, new_y)
        xy_symmetry_functions = {
            "noop":   lambda (x, y): (x, y),
            "rot90":  lambda (x, y): (y, self.size-x),
            "rot180": lambda (x, y): (self.size-x, self.size-y),
            "rot270": lambda (x, y): (self.size-y, x),
            "mirror-lr": lambda (x, y): (self.size-x, y),
            "mirror-ud": lambda (x, y): (x, self.size-y),
            "mirror-\\": lambda (x, y): (y, x),
            "mirror-/":  lambda (x, y): (self.size-y, self.size-x)
        }

        def update_ko_history(copy, name):
            if copy.ko is not None:
                copy.ko = xy_symmetry_functions[name](copy.ko)
            copy.history = [xy_symmetry_functions[name](a) if a is not
                            PASS_MOVE else PASS_MOVE for a in copy.history]

        copies = [self.copy() for i in range(8)]
        # copies[0] is the original.
        # rotate CCW 90
        copies[1].board = np.rot90(self.board, 1)
        update_ko_history(copies[1], "rot90")
        # rotate 180
        copies[2].board = np.rot90(self.board, 2)
        update_ko_history(copies[2], "rot180")
        # rotate CCW 270
        copies[3].board = np.rot90(self.board, 3)
        update_ko_history(copies[3], "rot270")
        # mirror left-right
        copies[4].board = np.fliplr(self.board)
        update_ko_history(copies[4], "mirror-lr")
        # mirror up-down
        copies[5].board = np.flipud(self.board)
        update_ko_history(copies[5], "mirror-ud")
        # mirror \ diagonal
        copies[6].board = np.transpose(self.board)
        update_ko_history(copies[6], "mirror-\\")
        # mirror / diagonal (equivalently: rotate 90 CCW then flip LR)
        copies[7].board = np.fliplr(copies[1].board)
        update_ko_history(copies[7], "mirror-/")
        return copies

    def from_sgf(self, sgf_string):
        raise NotImplementedError()

    def to_sgf(self, sgf_string):
        raise NotImplementedError()


class IllegalMove(Exception):
    pass