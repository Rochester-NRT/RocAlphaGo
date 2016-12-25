            shuffle_indices = np.load(f)
        if args.verbose:
            print "loading previous data shuffling indices"
    else:
        # create shuffled indices
        shuffle_indices = np.random.permutation(n_total_data)
        with open(shuffle_file, "w") as f:
            np.save(f, shuffle_indices)
        if args.verbose:
            print "created new data shuffling indices"
    # training indices are the first consecutive set of shuffled indices, val next, then test gets the remainder
    train_indices = shuffle_indices[0:n_train_data]
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    # test_indices = shuffle_indices[n_train_data + n_val_data:]

    # create dataset generators
    train_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["winners"],
        train_indices,
        args.minibatch,
        BOARD_TRANSFORMATIONS)
    val_data_generator = shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["winners"],
        val_indices,
        args.minibatch,
        BOARD_TRANSFORMATIONS)

    sgd = SGD(lr=args.learning_rate, decay=args.decay)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    samples_per_epoch = args.epoch_length or n_train_data

    if args.verbose:
        print "STARTING TRAINING"

    model.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=args.epochs,
        callbacks=[checkpointer, meta_writer],
        validation_data=val_data_generator,
        nb_val_samples=n_val_data or 1,  # Temporary hack
        show_accuracy=True,
        nb_worker=args.workers)

if __name__ == '__main__':
    run_training()
