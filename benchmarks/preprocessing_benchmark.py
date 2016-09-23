from AlphaGo.preprocessing.game_converter import game_converter
from cProfile import Profile

prof = Profile()

test_features = ["board", "turns_since", "liberties", "capture_size", "self_atari_size",
                 "liberties_after", "sensibleness", "zeros"]
gc = game_converter(test_features)
args = ('tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf', 19)


def run_convert_game():
    for traindata in gc.convert_game(*args):
        pass

prof.runcall(run_convert_game)
prof.dump_stats('bench_results.prof')
