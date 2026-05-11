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

    def __init__(self, particles, tree):
        self.particles = particles   # list of (model_idx, state)
        self.tree = tree
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
    Particle-based belief tree for belief space planning.

    possible_models: a fixed list of sampled world-model copies, shared across
    all nodes.  Each particle in a BeliefNode is (i, state) where i indexes
    into this list.
    """

    def __init__(self, world_model, initial_state_belief, num_model_particles):
        """
        initial_state_belief is a list of latent states that you might be in at the start.
        each node in the tree maintains a set of particles of size (num_model_particles * len(initial_state_belief)) by taking the cross product of possible_models and initial_state_belief.
        """
        self.possible_models = world_model.make_particle_based_approximation(num_model_particles)
        self.root = BeliefNode([(i, s) for i, model in enumerate(self.possible_models) for s in initial_state_belief], self)


    def expand(self, node, action, add_to_tree=True):
        """
        Sample new belief given current belief and action.
        By default, also adds to tree.
        """
        true_idx, true_state = random.choice(node.particles)
        true_model = self.possible_models[true_idx]
        true_next_state = true_model.sample_next_state(true_state, action)
        observation = true_model.sample_observation_given_state(true_next_state)

        # check if child already exists
        if node.get_child(action, observation) is not None:
            return observation, node.get_child(action, observation)
        
        # create and add the node
        # as a heuristic we always add (true_idx, true_next_state) as a particle in the new node, to ensure that the sampled observation is possible under the new node's particles
        # technically dont have to do this in large particle limit but it seems reasonable as a biasing heuristic to ensure we dont get all tiny-weight particles

        log_weights = []
        particles = []
        for idx, state in node.particles:
            if len(particles) == 0:
                # first one is copy of what we know work
                model = self.possible_models[true_idx]
                next_state = true_next_state
            else:
                model = self.possible_models[idx]
                next_state = model.sample_next_state(state, action)
            log_weights.append(model.observation_log_likelihood_given_state(next_state, observation))
            particles.append((idx, next_state))

        weight_dist = Categorical({k: w for k, w in enumerate(log_weights)})
        log_obs_prob = weight_dist.Z.item() - math.log(len(node.particles))  # log p(observation | action, node)
        weight_dist = weight_dist.normalize()
        new_particles = [particles[weight_dist.sample()] for _ in particles]

        new_node = BeliefNode(new_particles, self)
        new_node.log_obs_prob = log_obs_prob  # store this for free while we have the weights around, since it can be useful for info-gain calculations
        if add_to_tree:
            node.add_child(action, observation, new_node)

        return observation, new_node

