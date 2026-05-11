"""
Belief-space expectimax tree via a particle-based belief tree.

A BeliefNode holds a cloud of (model_idx, state) particles, where model_idx
indexes into a fixed set of sampled world-model particles shared across the
whole tree.  Edges are (action, observation) pairs; expanding a node means
sampling an observation, resampling the particle cloud, and returning a child.
"""

import math
import random
import torch
from tqdm import tqdm
from distributions import Categorical


class BeliefNode:
    """
    Particle cloud over hidden states at one point in the tree.
    Note "hidden state" here could be model parameters, world state, or both (or something else)
    Just think of it as a bag of particles representing belief over a latent thing
    """

    def __init__(self, histogram, tree, history=()):
        self.histogram = histogram   # Categorical of model_idx
        self.tree = tree
        self.history = list(history)  # [obs, act, obs, act, ...] from root to this node
        self.children = {}           # action → [(observation, BeliefNode), ...]

    def get_child(self, action, observation):
        """Return the stored child for (action, observation) by equality, or None."""
        for obs, child in self.children.get(action, []):
            if obs == observation:
                return child
        return None

    def add_child(self, action, observation, child):
        """Add a child for (action, observation).  Error if one already exists by equality."""
        if action not in self.children:
            self.children[action] = list()
        for obs, existing_child in self.children[action]:
            if obs == observation:
                raise ValueError("Child already exists for this (action, observation) pair.")
        self.children[action].append((observation, child))



class BeliefTree:
    """
    Tree for belief space planning.

    possible_models: a fixed list of sampled world-model copies, shared across
    all nodes.  Each particle in a BeliefNode indexes into this list.
    """

    def __init__(self, world_model, initial_history, num_model_particles, exact_belief_updates=True):
        self.possible_models = world_model.make_particle_based_approximation(num_model_particles)

        histogram = Categorical({i: 1 for i in range(len(self.possible_models))})
        
        self.root = BeliefNode(histogram, self, initial_history)

        self.exact_belief_updates = exact_belief_updates

        assert len(initial_history) % 2 == 1, "History should be in the form [obs, act, obs, act, ..., obs]"


    def expand(self, node, action, add_to_tree=True):
        """
        Sample new belief given current belief and action.
        By default, also adds to tree.
        """
        true_idx = node.histogram.normalize().sample()
        true_model = self.possible_models[true_idx]
        history = node.history
        observations, actions = history[::2], history[1::2]
        next_observation = true_model.sample_observation(observations, actions + [action])

        # check if child already exists
        if node.get_child(action, next_observation) is not None:
            return next_observation, node.get_child(action, next_observation)

        # create and add the node
        # for each model in the belief, compute likelihood of next_observation given action and past history, to get a distribution over models for the new node
        likelihoods = {i: self.possible_models[i].observation_log_likelihood(next_observation, observations, actions + [action])
                       for i in node.histogram.keys()}
        
        # scale by likelihoods to get exact posterior
        new_histogram = Categorical({i: node.histogram.logits[i] + likelihoods[i] for i in node.histogram.keys()}).normalize()

        if not self.exact_belief_updates:
            # resample
            samples = [new_histogram.sample() for _ in range(len(self.possible_models))]
            new_histogram = Categorical({i: math.log(samples.count(i)) for i in set(samples)}).normalize()

        new_node = BeliefNode(new_histogram, self, history=node.history + [action, next_observation])
        
        if add_to_tree:
            node.add_child(action, next_observation, new_node)

        return next_observation, new_node
