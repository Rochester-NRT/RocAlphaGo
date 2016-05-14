import numpy as np
from AlphaGo.expert import Expert
from AlphaGo.go import GameState

WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None

class UserInfo(object):

    """
    Indicate the user info which is used in Smart Game Format.
    """

    def __init__(self, name, rank):
        """
        Constructor

        Args:
            name (str): the name of the player
            rank (str): the ranking expression of the player
        """
        self.name = name
        self.rank = rank


class GameStateManager(object):

    """
    manage GameState operation, and take the signal from the front-end
    """

    def __init__(self, policyNet):
        self.game_state_instance = GameState()
        self.black_player = None
        self.white_player = None
        self.policy = policyNet
        self.expert = Expert(self.policy, self.game_state_instance)
        self.lastMove = (-1, -1)

    def _is_first_step(self):
        return (not self.black_player) and (not self.white_player)

    def _is_second_step(self):
        return (self.black_player) and (not self.white_player)

    def _computer_move(self):
        nextMove = self.expert.mcts_getMove(self.game_state_instance, self.lastMove);
        return nextMove

    def _pack_computer_move(self, x, y):
        return {
            'x': x,
            'y': y,
            'name': 'computer',
            'rank': '99'
        }

    def do_workflow(self, bundle):
        """
        Main workflow, entry point.

        Args:
            bundle (dict):
                {
                    "sessionId": (int)
                    "x": (int),
                    "y": (int),
                    "rank": (str),
                    "name": (str)
                }
        Returns:
            step response (dict):
                {
                    "x": (int),
                    "y": (int),
                    "legal_moves": [(tuple(2))],
                    "dead_stones": [(tuple(2))]
                }
        Raises:
            pass
        """
        if not (bundle['x'], bundle['y']) == (-1, -1):
            self.perform_step(bundle)
            human_remove_set = self.game_state_instance.last_remove_set.copy()
        else:
            human_remove_set = set()
            self.expert.aiColor = BLACK

        #computer need to know lastMove
        self.lastMove = (bundle['x'], bundle['y'])
        computer_move = self._computer_move()
        computer_bundle = self._pack_computer_move(computer_move[0], computer_move[1])
        self.perform_step(computer_bundle)
        computer_remove_set = self.game_state_instance.last_remove_set.copy()

        if 'sessionId' in bundle:
            sessionId = bundle['sessionId']
        else:
            sessionId = "xxx"

        return {
            'x': computer_bundle['x'],
            'y': computer_bundle['y'],
            'legal_moves': self.game_state_instance.get_legal_moves(),
            'dead_stones': list(set(list(human_remove_set) + list(computer_remove_set))),
            'sessionId': sessionId
        }

    def perform_step(self, bundle):
        """
        Perform a step to GameState object, input could be from either computer or human.

        Args:
            bundle (dict):
                {
                    "x": (int),
                    "y": (int),
                    "rank": (str),
                    "name": (str)
                }
        Returns:
            None
        Raises:
            pass
        """

        if self._is_first_step(): 
            self.black_player = UserInfo(bundle['name'], bundle['rank'])
        elif self._is_second_step(): 
            self.white_player = UserInfo(bundle['name'], bundle['rank'])

        self.game_state_instance.do_move((bundle['x'], bundle['y']))

    def print_board(self):
        board = self.game_state_instance.board

        for y in range(self.game_state_instance.size):
            row = ""
            for x in range(self.game_state_instance.size):
                mapping = {
                    1: 'B',
                    -1: 'W',
                    0: '.'
                }
                row += mapping[int(board[x][y])] + ' '
            print row
        