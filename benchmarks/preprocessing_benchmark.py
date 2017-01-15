import os, sys
p = os.path
parentddir = p.abspath(p.join(p.dirname(__file__), ".."))
sys.path.append(parentddir)

from AlphaGo.preprocessing.game_converter import GameConverter
from cProfile import Profile

prof = Profile()

test_features = ["board", "turns_since", "liberties", "capture_size", "self_atari_size",
                 "liberties_after", "sensibleness", "zeros"]
gc = GameConverter(test_features)
args = ('tests/test_data/sgf/Lee-Sedol-vs-AlphaGo-20160309.sgf', 19)


def run_convert_game():
    for traindata in gc.convert_game(*args):
        pass

prof.runcall(run_convert_game)
prof.dump_stats('bench_results.prof')
