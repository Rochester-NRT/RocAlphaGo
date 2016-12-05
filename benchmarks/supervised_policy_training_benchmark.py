from AlphaGo.training.supervised_policy_trainer import run_training
from AlphaGo.models.policy import CNNPolicy
from cProfile import Profile

architecture = {'filters_per_layer': 128, 'layers': 12}
features = ['board', 'ones', 'turns_since']
policy = CNNPolicy(features, **architecture)
policy.save_model('model.json')

profile = Profile()

# --epochs 5 --minibatch 32 --learning-rate 0.01
arguments = ('model.json', 'debug_feature_planes.hdf5', 'training_results/', 5, 32, .01)


def run_supervised_policy_training():
    run_training(*arguments)


profile.runcall(run_supervised_policy_training)
profile.dump_stats('supervised_policy_training_bench_results.prof')
