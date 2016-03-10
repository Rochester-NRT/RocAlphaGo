from ai import GreedyPolicyPlayer
from models.policy import CNNPolicy
from interface.Play import play_match


features = ["board", "ones", "turns_since", "liberties", "capture_size",
            "self_atari_size", "liberties_after",
            "sensibleness", "zeros"]
policy1 = CNNPolicy(features)
policy2 = CNNPolicy(features)
player1 = GreedyPolicyPlayer(policy1)
player2 = GreedyPolicyPlayer(policy2)

match = play_match(player1, player2, 'test')
while True:
    end_of_game = match.play()
    print "turns played:", match.state.turns_played
    if end_of_game:
        break
    # End game prematurely for profiling purposes
    # elif match.state.turns_played > 8:
    #     break
print "board: \n", match.state.board
