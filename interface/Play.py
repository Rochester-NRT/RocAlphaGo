"""Interface for AlphaGo self-play"""
from AlphaGo.go import GameState


class play_match(object):
    """Interface to handle play between two players."""

    def __init__(self, player1, player2, save_dir=None, size=19):
        # super(ClassName, self).__init__()
        self.player1 = player1
        self.player2 = player2
        self.state = GameState(size=size)
        # I Propose that GameState should take a top-level save directory,
        # then automatically generate the specific file name

    def _play(self, player):
        move = player.get_move(self.state)
        # TODO: Fix is_eye?
        self.state.do_move(move)  # Return max prob sensible legal move
        # self.state.write_to_disk()
        if len(self.state.history) > 1:
            if self.state.history[-1] is None and self.state.history[-2] is None \
                    and self.state.current_player == -1:
                end_of_game = True
            else:
                end_of_game = False
        else:
            end_of_game = False
        return end_of_game

    def play(self):
        """Play one turn, update game state, save to disk"""
        end_of_game = self._play(self.player1)
        # This is incorrect.
        return end_of_game
