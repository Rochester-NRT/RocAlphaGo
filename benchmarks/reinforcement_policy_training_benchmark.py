from AlphaGo.training.reinforcement_policy_trainer import run_training
from AlphaGo.models.policy import CNNPolicy
import os
from cProfile import Profile

# make a miniature model for playing on a miniature 7x7 board
architecture = {'filters_per_layer': 32, 'layers': 4, 'board': 7}
features = ['board', 'ones', 'turns_since', 'liberties', 'capture_size',
            'self_atari_size', 'liberties_after', 'sensibleness']
policy = CNNPolicy(features, **architecture)

datadir = os.path.join('benchmarks', 'data')
modelfile = os.path.join(datadir, 'mini_rl_model.json')
weights = os.path.join(datadir, 'init_weights.hdf5')
outdir = os.path.join(datadir, 'rl_output')
stats_file = os.path.join(datadir, 'reinforcement_policy_trainer.prof')

if not os.path.exists(datadir):
    os.makedirs(datadir)
if not os.path.exists(weights):
    policy.model.save_weights(weights)
policy.save_model(modelfile)

profile = Profile()
arguments = (modelfile, weights, outdir, '--learning-rate', '0.001', '--save-every', '2',
             '--game-batch', '20', '--iterations', '10', '--verbose')

profile.runcall(run_training, arguments)
profile.dump_stats(stats_file)
