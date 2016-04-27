﻿
from AlphaGo.go import GameState
from AlphaGo.mcts import MCTS
from AlphaGo.mcts import TreeNode
import random
import unittest

import os
import argparse
import json
import cPickle as pickle
#import six.moves.cPickle as pickle
import random
import numpy as np
from AlphaGo.models.policy import CNNPolicy

from random import shuffle
import uuid
from AlphaGo import util
import re
import subprocess

WHITE = -1
BLACK = +1
EMPTY = 0
PASS_MOVE = None

def policy_network_random_noEyes(state):
	moves = state.get_legal_moves(include_eyes=False)
	# 'random' distribution over positions that is smallest
	# at (0,0) and largest at (18,18)
	probs = np.arange(361, dtype=np.float)
	probs = probs / probs.sum()
	return zip(moves, probs)

def policy_network_random_withEyes(state):
	moves = state.get_legal_moves(include_eyes=True)
	# 'random' distribution over positions that is smallest
	# at (0,0) and largest at (18,18)
	probs = np.arange(361, dtype=np.float)
	probs = probs / probs.sum()
	return zip(moves, probs)

def policy_network_random(state):
	moves = state.get_legal_moves()
	actions = []
	for move in moves:
		actions.append((move, random.uniform(0, 1)))
	return actions


class Expert():

	def __init__(self, policy, state, aiColor = WHITE):
		self.policy = policy
		self.mcts = MCTS(state, self.value_network, self.policy_network, self.rollout_policy, n_search=10, c_puct = 5, playout_depth = 1, rollout_limit = 10)
		self.aiColor = aiColor
	def mcts_getMove(self, state, lastAction):
		if lastAction[0] != -1:
			self.mcts.update_with_move(lastAction)

		#quick move for first 10 steps
		if len(state.history) < 4:
			moves = self.policy_network(state);
			move = moves[0][0];
		else:
			move = self.mcts.get_move(state)

		self.mcts.update_with_move(move)
		return move

		## update the tree		
		#if lastAction[0] != -1:
		#	if lastAction not in self.mcts.treenode.children:
		#		self.mcts.treenode.expansion([(lastAction, 0.5)])
		#		self.mcts.treenode.updateU_value([(lastAction, 0.5)])
		#	self.mcts.state.do_move(lastAction)
		#	self.mcts.treenode = self.mcts.treenode.children[lastAction]

		#nextAction = self.mcts.getMove(3, 10)
		#self.mcts.state.do_move(nextAction)

		#if nextAction not in self.mcts.treenode.children:
		#	self.mcts.treenode.expansion([(nextAction, 0.5)])
		#	self.mcts.treenode.updateU_value([(nextAction, 0.5)])
		#self.mcts.treenode = self.mcts.treenode.children[nextAction] #drop previous root node

	def policy_network(self, state):
	    nextMoveList = self.policy.eval_state(state, state.get_legal_moves())
	    srtList = sorted(nextMoveList, key=lambda probDistribution: probDistribution[1], reverse=True);
	    res = srtList[0:30]
	    #shuffle(res)
	    return res


	def value_network(self, state):
		sgfId = str(uuid.uuid4())
		sgfPath = "d:\\tmp\\%s.value_network.sgf" % sgfId;
		util.gamestate_to_sgf(state,  sgfPath)
		t = subprocess.check_output(["cmd.exe", " /c D:\ps\club\Go\geekgo\eval.bat %s" % sgfPath])
		m = re.search('B:(\d+);W:(\d+)', t)
		blackImpactScope = float(m.group(1));
		whiteImpactScope = float(m.group(2));
		value = blackImpactScope / (blackImpactScope + whiteImpactScope)
		if self.aiColor != state.current_player:
			value = 1 - value
		value = value * 999999
		print (value, state.history[-5:-1])
		return value

	def rollout_policy(self, state):
		return policy_network_random_noEyes(state)

	def rollout_policy_score(state):
		nDepth = 3;
		#let you go first
		numOfCaptured = 0;
		numOfBeCaptured = 0;
		yourTurn = True
		#Assume AI always take WHITE
		if state.current_player == BLACK:
			yourTurn = False
		for i in range(0, nDepth - 1):
			nextMoveList = policy_network_random(state)
			state.do_move(nextMoveList[0][0])
			if yourTurn:
				numOfCaptured += len(state.last_remove_set)
			else:
				numOfBeCaptured += len(state.last_remove_set)

			yourTurn = not yourTurn
		return numOfCaptured / (numOfBeCaptured + 10)
