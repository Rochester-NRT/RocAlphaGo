import os, argparse
import numpy as np
import game_logic as gl
from sgflib.sgflib import SGFParser, GameTreeEndError

class game_converter:
    def __init__(self,target_format="deep"):
        self.index_at = {'a':0,'b':1,'c':2,'d':3,
                         'e':4,'f':5,'g':6,'h':7,
                         'i':8,'j':9,'k':10,'l':11,
                         'm':12,'n':13,'o':14,'p':15,
                         'q':16,'r':17,'s':18}
        self.target_format = target_format # todo: implement this

    # convert move into board indices
    def parse_raw_move(self,raw_move):
        pos = list(str(raw_move)[3:5])
        col = self.index_at[pos[0]]
        row = self.index_at[pos[1]]
        move = {'col':col,'row':row}
        return move

    # convert indices into 19x19 training label
    def encode_label(self,move):
        # convert move to one-hot encoding
        one_hot = np.zeros((19,19),dtype=bool)
        one_hot[move['col']][move['row']] = 1
        return one_hot

    # prepare training sample
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
        state[0][move['col']][move['row']] = 1
        state[2][move['col']][move['row']] = 0

        # compute feature slices based on game logic
        gl.check_for_capture(state[0:2])
        gl.update_move_ages(state[4:12],move)
        gl.update_current_liberties(state[0:2],state[12:20])
        gl.update_capture_sizes(state[0:2],state[20:29])
        gl.update_self_atari_sizes(state[0:2],state[29:37])
        gl.update_future_liberties(state[0:2],state[37:44])
        gl.update_ladder_captures(state[0:2],state[44])
        gl.update_ladder_escapes(state[0:2],state[45])
        gl.update_sensibleness(state[0:2],state[46])

    # convert full game into training samples
    def convert_game(self,file_name):
        with open(file_name,'r') as file_object:
            sgf_object = SGFParser(file_object.read())
        c = sgf_object.parse().cursor()
        states = []
        actions = []
        while True:
            try:
                move = self.parse_raw_move(c.next())
                actions.append(self.encode_label(move))
                self.append_state(states,move)
            except GameTreeEndError:
                # remove last board state since it has no label
                states = states[0:-1]
                break
        return zip(states, actions)

    # lazily convert folder of games into training samples
    def batch_convert(self,folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            print file_name
            training_samples = self.convert_game(os.path.join(folder_path,file_name))
            for sample in training_samples:
                yield sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a folder of Go game files for training our neural network model.')
    parser.add_argument("infolder", help="Relative path to folder containing games")
    parser.add_argument("outfolder", help="Relative path to target folder. Will be created if it does not exist.")
    parser.add_argument("-t","--target_format", help="One of: 'deep', 'shallow', or 'value'. Defaults to 'deep'")
    args = parser.parse_args()

    if not args.target_format: target_format = "deep"
    elif any([args.target_format == t for t in ["deep","shallow","value"]]):
        target_format = args.target_format
    else: raise ValueError("Unrecognized target format")

    converter = game_converter(target_format)

    for sample in converter.batch_convert(args.infolder):
        # sample[0] is the state  (48x19x19 feature tensor)
        # sample[1] is the action (19x19 class label)
        # write to args.outfolder
        x,y = sample
    print np.sum(x[4:12].astype(int),axis=0)
