from AlphaGo.models.game_converter import game_converter
from cProfile import Profile

prof = Profile()

test_features = ["board", "turns_since", "liberties", "capture_size", "self_atari_size", "liberties_after", "sensibleness", "zeros"]
gc = game_converter()
args = ('tests/test_sgfs/AlphaGo-vs-Lee-Sedol-20160310-first10only.sgf', test_features)

prof.runcall(gc.convert_game, *args)
prof.dump_stats('bench_results.prof')
