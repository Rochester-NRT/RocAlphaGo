#!/usr/bin/env python
# standard libraries
import os
import json
import uuid

# relevant third-party libraries
import bottle
from bottle import request, response, route
from keras.callbacks import ModelCheckpoint

# local modules
from AlphaGo.models.policy import CNNPolicy
import go
import gsm as GameStateMan

# global variables
DEBUG = True
GSM_POOL = dict()
POLICY = None


def init_cnnpolicynetwork():
    global POLICY

    train_folder = '/home/yimlin/betago_workspace/deploy'
    #train_folder = 'D:\ps\club\Go\models'
    metapath = os.path.join(train_folder, 'all_feat_model.json')

    with open(metapath) as metafile:
        metadata = json.load(metafile)

    weights_file = os.path.join(train_folder, 'weights.models_redoSL-continueOn8thEpoch_20160509.hdf5')
    arch = {'filters_per_layer': 128, 'layers': 12} # args to CNNPolicy.create_network()
    POLICY = CNNPolicy(feature_list=metadata['feature_list'], **arch)
    POLICY.model.load_weights(weights_file)
    #POLICY.model.compile(loss='categorical_crossentropy', optimizer='sgd')

init_cnnpolicynetwork()


@route('/move', method='OPTIONS')
@route('/move', method='POST')
def move():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'origin, x-csrftoken, content-type, accept'

    if request.method == 'OPTIONS':
        return 'received options request'

    incoming_bundle = request.json

    # determine what session id to use
    if ('sessionId' in incoming_bundle and (incoming_bundle['sessionId'] !="" )):
        session_id = incoming_bundle['sessionId']
        print session_id
    else:
        session_id = str(uuid.uuid4())
    incoming_bundle['sessionId'] = session_id
    print session_id

    # retrieve or create GameStateManager
    if session_id not in GSM_POOL:
        gsm = GameStateMan.GameStateManager(POLICY)
        GSM_POOL[session_id] = gsm
    else:
        gsm = GSM_POOL[session_id]

    resp = gsm.do_workflow(incoming_bundle)
    resp['sessionId'] = session_id

    if DEBUG:
        gsm.print_board()
        print 'input: ', str(request.body.read())
        print 'size_gsm_pool: ', len(GSM_POOL)

    return resp


if __name__ == "__main__":
    init_cnnpolicynetwork()
    bottle.run(host='0.0.0.0', port=80, debug=True)
