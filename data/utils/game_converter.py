from copy import copy, deepcopy
from sgflib.sgflib import SGFParser, GameTreeEndError

def convert(fileName): # only handles a single file for now
    with open(fileName,'r') as game_file:
        sgf_object = SGFParser(game_file.read())
        c = sgf_object.parse().cursor()
        s = state()
        while True:
            try: s.update(c.next())
            except GameTreeEndError: break
    print s

class state:
    ind = {'a':0,'b':1,'c':2,'d':3,
           'e':4,'f':5,'g':6,'h':7,
           'i':8,'j':9,'k':10,'l':11,
           'm':12,'n':13,'o':14,'p':15,
           'q':16,'r':17,'s':18}

    def __init__(self):
        self.board = [[0 for y in range(19)] for x in range(19)]
        self.history = []

    def update(self, move):
        color = 1 if str(move)[1] == 'W' else 2
        pos = list(str(move)[3:5])
        x = self.ind[pos[0]]
        y = self.ind[pos[1]]
        self.board[x][y] = color
        self.history.append([deepcopy(board_state) for board_state in self.board])

    def __str__(self):
        s = ""
        for i in range(len(self.history)):
            for x in range(19):
                for y in range(19):
                    s += str(self.history[i][x][y]) + " "
                s += "\n"
            s += "\n"
        return s

if __name__ == '__main__':
    # TODO: Update to accept batch command line arguments
    convert("../SGF/friday_tournament.sgf")
