import numpy as np
import os
import h5py as h5
import json
from AlphaGo.models.policy import CNNPolicy

def run_android( cmd_line_args = None ):

    """Create tensorflow .pb model file. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser( description = 'Create tensorflow .pb model file.' )
    # required args
    parser.add_argument( "model",           help= "Path to a JSON model file (i.e. from CNNPolicy.save_model())" )
    parser.add_argument( "out_directory",   help= "directory where metadata and weights will be saved" )
    parser.add_argument( "weights",         help= "Name of a .h5 weights file (in the output directory) to load to resume training" )
    # optional
    parser.add_argument( "--verbose", "-v", help= "Turn on verbose mode", default = False, action = "store_true" )

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args( cmd_line_args )

    if args.verbose:
        print( "trying to load model from %s with weights %s" %
                  ( args.out_directory, os.path.join( args.out_directory, args.weights ) ) )
        
    import tensorflow  as tf
    with tf.Session() as sess:

        from keras import backend as K
        K.set_session(sess)

        # load model from json spec
        policy         = CNNPolicy.load_model( args.model )
        model_features = policy.preprocessor.feature_list
        model          = policy.model
        model.load_weights( os.path.join( args.out_directory, args.weights ) )

        # count amount of parameters in model
        total_parameters = 0
        for variable in tf.trainable_variables():

            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print( shape )
            # print( len( shape ) )

            variable_parametes = 1
            for dim in shape:

                # print( dim )
                variable_parametes *= dim.value

            # print( variable_parametes )
            total_parameters += variable_parametes
        print( "parameters: " + str( total_parameters ) )

        # import used to get only relevant model data
        from tensorflow.python.framework.graph_util import convert_variables_to_constants

        # show names of all layers
        if args.verbose:
            print( "All tensors/variables" )
            print[n.name for n in tf.get_default_graph().as_graph_def().node]
            # extra whiteline
            print " \n"

        # get all layes necessary to compute output layer 'Softmax'
        minimal_graph = convert_variables_to_constants( sess, sess.graph_def, ["Softmax"] )
        
        # show names in minimal_graph

        if args.verbose:
            print( "All tensors/variables in minimal_graph" )
            print[n.name for n in minimal_graph.node ]
        
        # create .pb file
        tf.train.write_graph( minimal_graph, '.', 'minimal_graph.pb', as_text = False )

if __name__ == '__main__':
    run_android()
