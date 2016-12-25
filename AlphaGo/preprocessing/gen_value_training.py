    while True:  # n in xrange(n_training_pairs / batch_size):
        X, winners = play_batch(player_RL, player_SL, batch_size, features)
        if X is not None:
            try:
                h5_states.resize((next_idx + batch_size, n_features, bd_size, bd_size))
                h5_winners.resize((next_idx + batch_size, 1))
                h5_states[next_idx:] = X
                h5_winners[next_idx:] = winners
                next_idx += batch_size
            except Exception as e:
                warnings.warn("Unknown error occured during batch save to HDF5 file: {}".format(out_pth))
                raise e
        n_pairs += 1
        if n_pairs >= n_training_pairs / batch_size:
            break
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Play games used for training'
                                     'value network (third phase of pipeline). '
                                     'The final policy from the RL phase plays '
                                     'against itself and training pairs for value '
                                     'network are generated from the outcome in each '
                                     'games, following an off-policy, uniform random move')
    parser.add_argument("SL_weights_path", help="Path to file with supervised "
                        "learning policy weights.")
    parser.add_argument("RL_weights_path", help="Path to file with reinforcement "
                        "learning policy weights.")
    parser.add_argument("model_path", help="Path to network architecture file.")
    parser.add_argument("--out_pth", "-o", help="Path to where the training "
                        "pairs will be saved. Default: None", default=None)
    parser.add_argument("--load_from_file", help="Path to HDF5 file to continue from."
                        " Default: None", default=None)
    parser.add_argument("--n_training_pairs", help="Number of training pairs to generate. "
        "(Default: 10)", type=int, default=10)
    parser.add_argument("--batch_size", help="Number of games to run in parallel. "
        "(Default: 2)", type=int, default=2)
    parser.add_argument("--board_size", help="Board size (int). "
        "(Default: 19)", type=int, default=19)
    args = parser.parse_args()

    # Load architecture and weights from file
    policy_SL = CNNPolicy.load_model(args.model_path)
    features = policy_SL.preprocessor.feature_list
    if "color" not in features:
        features.append("color")
    policy_SL.model.load_weights(args.SL_weights_path)
    policy_RL = CNNPolicy.load_model(args.model_path)
    policy_RL.model.load_weights(args.RL_weights_path)
    # Create player object that plays against itself (for both RL and SL phases)
    player_RL = ProbabilisticPolicyPlayer(policy_RL)
    player_SL = ProbabilisticPolicyPlayer(policy_SL)
    run(player_RL, player_SL, args.out_pth, args.n_training_pairs,
        args.batch_size, args.board_size, features)