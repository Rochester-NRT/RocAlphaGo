"""Interface for AlphaGo self-play"""
from go import GameState


class play_match(object):
    """Interface to handle play between two players."""
    def __init__(self, player1, player2, save_dir, size=19):
        # super(ClassName, self).__init__()
        self.player1 = player1
        self.player2 = player2
        # self.state = GameState(save_dir, size=size)
        self.state = GameState(size=size)
        # I Propose that GameState should take a top-level save directory,
        # then automatically generate the specific file name

    def _play(self, player):
        move = player.get_move(self.state)
        end_of_game = self.state.do_move(move)
        # self.state.write_to_disk()
        return end_of_game

    def play(self):
        """Play one turn for each player, update game state, save to disk"""
        end_of_game = self._play(self.player1)
        if not end_of_game:
            end_of_game = self._play(self.player2)
        return end_of_game


########### Example implementation #####################
# from Play import play_match


# save_dir = 'path/to/game_state/save_dir'
# match = play_match(alphago, pachi, save_dir)
# while True:
#     end_of_game = match.play()
#     if end_of_game:
#         break
