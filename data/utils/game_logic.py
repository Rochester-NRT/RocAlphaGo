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

# @param stones: 19x19 board consist of [0,1,2] denote [empty, black, white]
# Output: curr_liberties: 8x19x19 boolean:
### An index of a slice is 1 iff the position has that many liberties.
def update_current_liberties(stones):
    # Function to find the liberty of one stone
    def liberty_count(i, j):
        q=0 #liberty count
        if stones[i+1][j] == 0:
            q = q + 1
        if stones[i][j+1] == 0:
            q = q + 1
        if i-1 > 0 and stones[i-1][j] == 0:
            q = q + 1
        if j -1 > 0 and stones[i][j -1 ] == 0:
            q = q + 1
        return q
    # Record the liberty position
    def liberty_pos(i, j):
        pos=[]
        if stones[i+1][j] == 0:
            pos.append([i+1, j])
        if stones[i][j+1] == 0:
            pos.append([i, j+1])
        if i - 1 >= 0 and stones[i-1][j] == 0:
            pos.append([i-1, j])
        if j - 1 >= 0 and stones[i][j-1] == 0:
            pos.append([i, j-1])
        return pos

     # Scanning through the board
    lib_considered=[]
    for i in range(0, 19):
        for j in range(0, 19):
            if [i, j] == [x for x in lib_considered]:
                continue

            # The first position picked
            lib_considered.append(i, j)
            lib_set = [i, j]
            lib_c = liberty_count(i, j)
            lib_set.append(liberty_pos(i, j)

            # Scanning through 4 directions to find the same color cluster
            while stone[i][j] == stone[i][j+1]:
                lib_set.append(liberty_pos(i, j+1))
                lib_c = lib_c + liberty_count(i, j+1)
                j = j + 1
            
            while stone[i][j] == stone[i+1][j]:
                lib_set.append(liberty_pos(i+1, j))
                lib_c = lib_c + liberty_count(i+1, j)
                i = i + 1

            while i - 1 >= 0 and stone[i][j] == stone[i-1][j]:
                lib_set.append(liberty_pos(i-1, j)
                lib_c = lib_c + liberty_count(i-1, j)
                i = i - 1

            while j - 1 >= 0 and stone[i][j] == stone[i][j-1]:
                lib_set.append(liberty_pos(i, j-1))
                lib_c = lib_c + liberty_count(i, j-1)
                j = j - 1

            # Combine the liberty position of the cluster found
            lib_set = set(lib_set)

            # Update onehot encoding rectangular prisim
            if lib_c > 0 and lib_c < 8:
                for pos in lib_set:
                    curr_liberties[lib_c-1][pos[0]][pos[1]] = 1
            elif lib_c >= 8:
                for pos in lib_set:
                    curr_liberties[7][pos[0]][pos[1]] = 1

    return curr_liberties

# @param capture_sizes: 8x19x19 boolean:
### An index of a slice is 1 iff a move there would capture that many opponents.
def update_capture_sizes(stones,capture_sizes):
    pass

# @param self_ataris: 8x19x19 boolean:
### An index of a slice is 1 iff the playing a move there would capture that many of player's own stones.
def update_self_atari_sizes(stones,self_ataris):
    pass

# @param stones: 19x19 board consist of [0,1,2] denote [empty, black, white]
# @param move: dict containing the row,col index of the most recent move.
# Output: future_liberties: 8x19x19 boolean:
### An index of a slice is 1 iff playing a move there would yield that many liberties.

def update_future_liberties(stones, move):
    # very similar to curr_liberties, only we do not scan the whole board this time
    # Only one cluster which contains the new move is considered
    i=move['row']
    j=move['col']

    lib_set = [i, j]
    lib_c = liberty_count(i, j)
    lib_set.append(liberty_pos(i, j)

    while stone[i][j] == stone[i][j+1]:
        lib_set.append(liberty_pos(i, j+1))
        lib_c = lib_c + liberty_count(i, j+1)
        j = j + 1
    
    while stone[i][j] == stone[i+1][j]:
        lib_set.append(liberty_pos(i+1, j))
        lib_c = lib_c + liberty_count(i+1, j)
        i = i + 1

    while i - 1 >= 0 and stone[i][j] == stone[i-1][j]:
        lib_set.append(liberty_pos(i-1, j)
        lib_c = lib_c + liberty_count(i-1, j)
        i = i - 1

    while j - 1 >= 0 and stone[i][j] == stone[i][j-1]:
        lib_set.append(liberty_pos(i, j-1))
        lib_c = lib_c + liberty_count(i, j-1)
        j = j - 1

    lib_set = set(lib_set)

    future_liberties = curr_liberties(stones)
    # read the old data, and then update
    if lib_c > 0 and lib_c < 8:
        for pos in lib_set:
            future_liberties[lib_c-1][pos[0]][pos[1]] = 1
    elif lib_c >= 8:
        for pos in lib_set:
            future_liberties[7][pos[0]][pos[1]] = 1

    return future_liberties


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
