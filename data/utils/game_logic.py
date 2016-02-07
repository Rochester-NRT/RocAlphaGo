import numpy as np

''' These methods implement the updates required
to prepare the features of each training tensor'''

# @param ages: 8x19x19 boolean:
### An index of a slice is 1 iff a move is that old.
# @param move: dict containing the row,col index of the most recent move.
def update_move_ages(ages,move):
    # final slice collects all moves older than 6 turns
    ages[7] = np.logical_or(ages[6], ages[7])
    # intermediate slices get shifted up
    ages[6] = ages[5]
    ages[5] = ages[4]
    ages[4] = ages[3]
    ages[3] = ages[2]
    ages[2] = ages[1]
    # youngest slice steals a 1 from the unplayed pool
    ages[1] = np.zeros((19,19),dtype=bool)
    ages[1][move['row']][move['col']] = 1
    ages[0][move['row']][move['col']] = 0

# @param stones: 3x19x19 boolean:
### The first slice has a 1 at an index if the current player has a stone there.
### The second slice has a 1 at an index if the current player's opponent has a stone there.
### The third slice has a 1 at an index if neither player has a stone there.
def check_for_capture(stones):
    pass

# @param curr_liberties: 8x19x19 boolean:
### An index of a slice is 1 iff the position has that many liberties.
def update_current_liberties(stones,curr_liberties):
    pass

# @param capture_sizes: 8x19x19 boolean:
### An index of a slice is 1 iff a move there would capture that many opponents.
def update_capture_sizes(stones,capture_sizes):
    pass

# @param self_ataris: 8x19x19 boolean:
### An index of a slice is 1 iff the playing a move there would capture that many of player's own stones.
def update_self_atari_sizes(stones,self_ataris):
    pass

# @param future_liberties: 8x19x19 boolean:
### An index of a slice is 1 iff playing a move there would yield that many liberties.
# @param stones: 19x19 board consist of [0,1,2] denote [empty, black, white]
# @param move: dict containing the row,col index of the most recent move.

def update_future_liberties(stones, move, future_liberties):
    # Getting the color of the new move
    color = stones[move['row']][move['col']]

    q=0 #liberty count
    if stones[move['row']+1][move['col']] == 0:
        q = q + 1
    if stones[move['row']][move['col']+1] == 0:
        q = q + 1
    if move['row']-1 > 0 and stones[move['row']-1][move['col']] == 0:
        q = q + 1
    if move['col'] -1 > 0 and stones[move['row']][move['col'] -1 ] == 0:
        q = q + 1

    future_liberties[q][move['row']][move['col']] = 1


# @param ladder_captures: 19x19 boolean:
### An index is 1 iff playing a move there would be a successful ladder capture.
def update_ladder_captures(stones,ladder_captures):
    pass

# @param ladder_escapes: 19x19 boolean:
### An index is 1 iff playing a move there would be a successful ladder escape.
def update_ladder_escapes(stones,ladder_escapes):
    pass

# @param sensibleness: 19x19 boolean:
### An index is 1 iff a move is legal and does not fill its own eyes.
def update_sensibleness(stones,sensibleness):
    pass
