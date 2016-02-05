import os, argparse
import numpy as np
from sgflib.sgflib import SGFParser, GameTreeEndError

class game_converter:
    def __init__(self):
        self.index_at = {'a':0,'b':1,'c':2,'d':3,
                         'e':4,'f':5,'g':6,'h':7,
                         'i':8,'j':9,'k':10,'l':11,
                         'm':12,'n':13,'o':14,'p':15,
                         'q':16,'r':17,'s':18}

    def parse_raw_move(self,raw_move):
        pos = list(str(raw_move)[3:5])
        row = self.index_at[pos[0]]
        col = self.index_at[pos[1]]
        return {'row':row,'col':col}

    def update_move_ages(self,age_slices,move):
        # final slice collects all moves older than 6 turns
        age_slices[7] = np.logical_or(age_slices[6], age_slices[7])
        # intermediate slices get shifted up
        age_slices[6] = age_slices[5]
        age_slices[5] = age_slices[4]
        age_slices[4] = age_slices[3]
        age_slices[3] = age_slices[2]
        age_slices[2] = age_slices[1]
        # youngest slice steals a 1 from the unplayed pool
        age_slices[1] = np.zeros((19,19),dtype=bool)
        age_slices[1][move['row']][move['col']] = 1
        age_slices[0][move['row']][move['col']] = 0

    def append_state(self,states,move):
        if len(states) is not 0:
            # copy last board state
            state = np.copy(states[-1])
            states.append(state)
            # relativise it to current player
            state[0:2] = state[0:2][::-1]
        else: # create board from scratch
            state = np.zeros((48,19,19),dtype=bool)
            # convert 3rd slice to ones because all board positions are empty
            # convert 4th slice to ones because it's a constant plane of ones
            # convert 5th slice to ones because no moves have been played yet
            state[2:5] = ~state[2:5]
            # add two states: 1 empty board and 1 in which we place 1st move
            states.append(state)
            state = np.copy(state)
            states.append(state)

        # perform move
        state[0][move['row']][move['col']] = 1
        state[2][move['row']][move['col']] = 0

        self.update_move_ages(state[4:12],move)

        # check_for_capture(states[0:2])
        # update_current_liberties(states[12:20])
        # update_capture_sizes(states[20:29])
        # update_self_atari_sizes(states[29:37])
        # update_future_liberties(states[37:44])
        # update_ladder_captures(states[44])
        # update_ladder_escapes(states[45])
        # update_sensibleness(states[46])

    def convert_game(self,file_name):
        with open(file_name,'r') as file_object:
            sgf_object = SGFParser(file_object.read())
        c = sgf_object.parse().cursor()
        states = []
        actions = []
        while True:
            try:
                move = self.parse_raw_move(c.next())
                actions.append(move)
                self.append_state(states,move)
            except GameTreeEndError: break
        return zip(states[0:-1], actions)

    def batch_convert(self,folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            training_samples = self.convert_game(os.path.join(folder_path,file_name))
            for sample in training_samples:
                yield sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a folder of Go game files for training our neural network model.')
    parser.add_argument("folder", help="Relative path to folder")
    parser.add_argument("target_format", help="Choose 'deep', 'shallow', or 'value'")
    args = parser.parse_args()

    c = game_converter()

    for sample in c.batch_convert(args.folder):
        # sample[0] is the state (3d binary feature representation of board)
        # sample[1] is the action (dict of row,col coordinates)
        pass
