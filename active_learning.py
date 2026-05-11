import math
from belief_tree import BeliefTree
from model import *

class ActiveLearner:

    def __init__(self, world_model, history, num_model_particles=20):
        self.world_model = world_model
        self.belief_tree = None
        self.num_model_particles = num_model_particles
        self.history = history

    def init_tree(self):
        """Initialize belief tree from the model's initial_belief."""
        self.belief_tree = BeliefTree(self.world_model, self.history, self.num_model_particles)

    def rollout_policy(self, policy, n_rollouts=1):
        """
        Roll out policy from the root, returning per-step info gains.

        policy(observations, actions) → action | None.
        If n_rollouts > 1, returns the per-step average across rollouts.
        """
        if n_rollouts > 1:
            rollouts = [self.rollout_policy(policy, n_rollouts=1) for _ in range(n_rollouts)]
            max_len = max(len(r) for r in rollouts)
            rollouts = [r + [0] * (max_len - len(r)) for r in rollouts]
            return [sum(r[i] for r in rollouts) / n_rollouts for i in range(max_len)]

        node = self.belief_tree.root
        observations, actions = [], []
        last_entropy = node.histogram.entropy()
        infogains = []
        while True:
            action = policy(observations, actions)
            if action is None:
                break
            actions.append(action)
            obs, node = self.belief_tree.expand(node, action, add_to_tree=False)
            observations.append(obs)
            new_entropy = node.histogram.entropy()
            infogains.append(last_entropy - new_entropy)
            last_entropy = new_entropy
        return infogains

