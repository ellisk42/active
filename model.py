"""
# world model hierarchy

WorldModel: abstract p(o_{t+1} | o_{1:t}, a_{1:t}) with a variational posterior over weights.
  Owns all variational parameters as a flat list of (prior, posterior) pairs.
POMDP: WorldModel that reifies latent state, using particle methods for inference.
PoE_POMDP: POMDP where p(s'|s,a) and p(o|s) are both products of experts.

# PoE_POMDP details

weights ~ P_prior
P(state_{t+1} | state_t, action_t, weights) = product of experts (using *dynamics rules*)
P(observation_t | state_t, weights) = product of experts (using *render rules*)

This model implicitly defines:
P(observations | actions, weights), by marginalizing out states
P(states | actions, observations, weights), which is approximated with particle smoothing. Expensive!

## implementation details

The state is essentially json, and each rule predicts a distribution over edits to the json.
When multiple rules try to edit the same part of the json, we form a PoE using their weights.
Rules can overwrite json list elements, and append to lists.
Deletion not implemented because you can hack it by overwriting with None.


# variational posterior: q_theta(weights)

maximize ELBO: E_{weights ~ q_theta} [ log P(observations | actions, weights) ] - KL(q_theta(weights) || P_prior(weights))

So need:
grad_theta E_{weights ~ q_theta} [ log P(observations | actions, weights) ]

Reparameterization lets us push the gradient inside the expectation, but we still need:
grad_weights log P(observations | actions, weights)

Which is
E_{states ~ P(states | actions, observations, weights)} [ grad_weights log P(observations, states | actions, weights) ]

P(states | actions, observations, weights) is approximated with particle smoothing.
"""
import copy
from abc import ABC, abstractmethod

import random
import math
import torch
from tqdm import tqdm
from distributions import *


class Expert:
    """Stateless rule: defines a transition or render distribution given an externally-supplied weight."""
    pass


class WorldModel(ABC):
    """Abstract distribution p(o_{t+1} | o_{1:t}, a_{1:t}) with a variational posterior over weights.

    parameter_distributions: list of (prior, posterior) pairs, one per learnable scalar weight.
    self.weights: current weight samples, kept in sync with parameter_distributions.
    """

    def __init__(self, parameter_distributions):
        self._parameter_distributions = list(parameter_distributions)
        self.weights = [post.sample() for _, post in self._parameter_distributions]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'weights':
                setattr(result, k, [w.detach().clone() if isinstance(w, torch.Tensor) else w
                                     for w in v])
            elif isinstance(v, torch.Tensor):
                setattr(result, k, v.detach().clone())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def pretty_print_observation(self, obs):
        # works assuming that observation is a 2D array of colors
        ANSI = {
            "black": "\033[40m",
            "grey":  "\033[47m",
            "green": "\033[42m",
        }
        RESET = "\033[0m"

        def render_cell(color):
            code = ANSI.get(color, ANSI["black"])
            char = " " if color == "black" else "X"
            return f"{code}{char}{RESET}"

        observation = list(zip(*obs))  # transpose
        for row in observation:
            print("".join(render_cell(cell) for cell in row))

    def parameter_distributions(self):
        return self._parameter_distributions

    def parameters(self):
        return [p for _, post in self._parameter_distributions for p in post.parameters()]

    def KL_Q_to_prior(self, n_samples=50):
        kl = 0
        for _ in range(n_samples):
            for prior, post in self._parameter_distributions:
                w = post.sample()
                kl += post.log_prob(w) - prior.log_prob(w)
        return kl / n_samples

    def resample_weights(self):
        self.weights = [post.sample() for _, post in self._parameter_distributions]

    @abstractmethod
    def observation_log_likelihood(self, observation, past_observations, past_actions, **kwargs):
        """log p(o_t | o_{1:t-1}, a_{1:t-1})"""
        pass

    def trajectory_log_likelihood(self, observations, actions, **kwargs):
        """log p(o_{1:T} | a_{1:T-1}).
        Default sums observation_log_likelihood over time steps; subclasses may override for efficiency."""
        total = 0
        for t in range(len(observations)):
            total += self.observation_log_likelihood(observations[t], observations[:t], actions[:t], **kwargs)
        return total

    @abstractmethod
    def sample_observation(self, past_observations, past_actions, **kwargs):
        """Sample o_{t+1} ~ p(o_{t+1} | o_{1:t}, a_{1:t})"""
        pass

    def elbo(self, observations, actions, num_weight_samples=1, **kwargs) -> dict:
        """Default ELBO: E_q[log p(o|w)] - KL(q||p), using reparameterized gradients.
        Assumes trajectory_log_likelihood is differentiable w.r.t. variational parameters.
        Overridden by POMDP, which needs the Fisher trick because its likelihood estimator
        (particle filter) has no reparameterizable gradient."""
        ll = 0
        kl = 0
        for _ in range(num_weight_samples):
            self.resample_weights()
            ll += self.trajectory_log_likelihood(observations, actions, **kwargs)
            kl += self.KL_Q_to_prior()
        ll /= num_weight_samples
        kl /= num_weight_samples
        return {'elbo': ll - kl, 'data_likelihood': ll.detach(), 'kl': kl.detach(), 'fisher': ll.detach()}

    def train(self, lr=0.01, iterations=100, dump_every=None, **elbo_kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elbo_curve = []
        fisher_curve = []
        kl_curve = []
        data_likelihood_curve = []
        pbar = tqdm(range(iterations))
        for iteration in pbar:
            optimizer.zero_grad()
            elbo = self.elbo(**elbo_kwargs)
            loss = -elbo['elbo']
            loss.backward()
            optimizer.step()
            elbo_curve.append(elbo['elbo'].item())
            fisher_curve.append(elbo['fisher'].item())
            kl_curve.append(elbo['kl'].item())
            data_likelihood_curve.append(elbo['data_likelihood'].item())
            pbar.set_postfix(elbo=f"{elbo['elbo'].item():.2f}", ll=f"{elbo['data_likelihood'].item():.2f}", kl=f"{elbo['kl'].item():.2f}")

            curves = {
                'elbo_curve': elbo_curve,
                'fisher_curve': fisher_curve,
                'kl_curve': kl_curve,
                'data_likelihood_curve': data_likelihood_curve,
            }
            if dump_every is not None and (iteration + 1) % dump_every == 0:
                yield iteration + 1, curves

        yield iterations, curves

    def make_particle_based_approximation(self, num_particles):
        """Returns num_particles deep-copied models, each with freshly resampled weights."""
        models = []
        for _ in range(num_particles):
            model = copy.deepcopy(self)
            model.resample_weights()
            models.append(model)
        return models


class POMDP(WorldModel):
    """WorldModel that reifies latent state.

    Subclasses must supply the transition kernel and observation model;
    this class provides particle-based implementations of the WorldModel API
    (observation_log_likelihood, sample_observation) and the POMDP inference
    routines (particle_filter, particle_smoothing_sampler, fisher_objective,
    elbo, plan_info_gain).
    """

    @abstractmethod
    def next_state_log_likelihood(self, state, next_state, action):
        """log p(next_state | state, action)"""
        pass

    @abstractmethod
    def sample_next_state(self, state, action):
        """Sample s' ~ p(s' | state, action)"""
        pass

    @abstractmethod
    def observation_log_likelihood_given_state(self, state, observation):
        """log p(observation | state)"""
        pass

    @abstractmethod
    def sample_observation_given_state(self, state):
        """Sample o ~ p(o | state)"""
        pass

    def propose_next_state(self, prev_state, prev_action, curr_observation):
        """Propose a next state given the previous state, action, and current observation.
        Used for particle inference; can be overridden to incorporate the current observation into the proposal.
        Returns (sample, log_prob of sample)"""
        proposal = self.sample_next_state(prev_state, prev_action)
        return proposal, None

    # --- WorldModel implementations ---

    def observation_log_likelihood(self, observation, past_observations, past_actions, initial_belief, num_particles):
        """Runs particle filter through past_observations/past_actions[:-1], then scores observation under past_actions[-1]."""
        belief_states, _ = self.particle_filter(initial_belief, past_observations, past_actions[:-1], num_particles)
        belief = belief_states[-1]
        particle_weights = []
        for particle in belief:
            next_particle = self.sample_next_state(particle, past_actions[-1])
            particle_weights.append(self.observation_log_likelihood_given_state(next_particle, observation))
        return torch.logsumexp(torch.tensor(particle_weights), dim=0) - math.log(num_particles)

    def trajectory_log_likelihood(self, observations, actions, initial_belief, num_particles):
        """Runs a single particle filter for efficiency rather than calling observation_log_likelihood per step."""
        return self.particle_filter(initial_belief, observations, actions, num_particles)[1]

    def sample_observation(self, past_observations, past_actions, initial_belief, num_particles=100):
        """Runs particle filter through past_observations/past_actions[:-1], then transitions with past_actions[-1]."""
        belief_states, _ = self.particle_filter(initial_belief, past_observations, past_actions[:-1], num_particles)
        state = random.choice(belief_states[-1])
        next_state = self.sample_next_state(state, past_actions[-1])
        return self.sample_observation_given_state(next_state)

    # --- POMDP inference routines ---

    def rollout(self, initial_state, actions):
        state = copy.deepcopy(initial_state)
        states = [state]
        observations = [self.sample_observation_given_state(state)]
        for action in actions:
            state = self.sample_next_state(state, action)
            states.append(state)
            observations.append(self.sample_observation_given_state(state))
        return states, observations

    def particle_filter(self, initial_belief, observations, actions, num_particles):
        """IMPORTANT! initial_belief is a list of states **for the first time step where there is an observation**
        We actually ignore the first observation..."""

        if len(initial_belief) < num_particles:
            initial_belief = initial_belief + random.choices(initial_belief, k=num_particles - len(initial_belief))

        belief_states = [initial_belief]

        # marginal likelihood of data
        logZ = 0.

        for obs, action in zip(observations[1:], actions):
            belief = belief_states[-1]
            new_belief = []
            particle_weights = []
            for particle in belief:
                new_particle, proposal_log_prob = self.propose_next_state(particle, action, obs)
                particle_weight = self.observation_log_likelihood_given_state(new_particle, obs)
                if proposal_log_prob is not None:
                    particle_weight -= proposal_log_prob
                    particle_weight += self.next_state_log_likelihood(particle, new_particle, action)
                particle_weights.append(particle_weight)
                new_belief.append((particle_weight, new_particle))
            # Normalize weights->build distribution
            resampling_distribution = Categorical({i: w for i, (w, _) in enumerate(new_belief)}).normalize()
            # Resample
            resampled_belief = [new_belief[resampling_distribution.sample()][1] for _ in range(num_particles)]
            belief_states.append(resampled_belief)

            # update logZ
            dZ = torch.logsumexp(torch.tensor(particle_weights), dim=0) - math.log(num_particles)
            logZ += dZ
        return belief_states, logZ

    def particle_smoothing_sampler(self, initial_belief, observations, actions, num_particles):
        """Yields samples from p(state_{1:T} | obs_{1:T}, actions_{1:T-1}) using a forward-filter backward-sample particle smoother."""
        belief_states, _ = self.particle_filter(initial_belief, observations, actions, num_particles)
        T = len(observations)

        assert len(belief_states) == T
        assert len(actions) == T - 1 # SUBTLE

        while True:
            trajectory = [random.sample(belief_states[-1], 1)[0]]  # sample from final belief
            for t in reversed(range(T - 1)):
                next_state = trajectory[0] # earliest state in trajectory so far is the "next state" for this step of backward sampling
                # Compute weights for backward sampling
                # FIXME: This could be precomputed outside the loop, which would pay off if you're taking tons of samples
                weights = []
                for particle in belief_states[t]:
                    try:
                        weights.append(self.next_state_log_likelihood(particle, next_state, actions[t]))
                    except Exception as e:
                        print(f"Error computing next_state_log_likelihood: {e}")
                        print(particle)
                        print(next_state)
                        print(actions[t])
                        print(self.transition_distribution(particle, actions[t]))
                        import pdb; pdb.set_trace()
                        
                # Normalize weights
                weight_dist = Categorical({i: w for i, w in enumerate(weights)}).normalize()
                # Sample previous state
                trajectory.insert(0, belief_states[t][weight_dist.sample()])
            yield trajectory

    def fisher_objective(self, observations, actions, initial_state, num_particles, num_smoothing_samples=1):
        """grad log(obs) =  grad E_{p(states|obs)} log p(obs,states)"""

        objective = 0

        num_state_samples = 0
        for trajectory in self.particle_smoothing_sampler(initial_belief=[initial_state], observations=observations, actions=actions, num_particles=num_particles):
            objective += sum(self.observation_log_likelihood_given_state(state, obs) for state, obs in zip(trajectory, observations))
            for t in range(len(trajectory) - 1):
                objective += self.next_state_log_likelihood(trajectory[t], trajectory[t + 1], actions[t])
            num_state_samples += 1
            if num_state_samples >= num_smoothing_samples:
                break

        return objective / num_state_samples

    def elbo(self, observations, actions, initial_state, num_particles, num_weight_samples, num_smoothing_samples=1):

        fisher = 0
        data_likelihood = 0
        kl = 0
        for _ in range(num_weight_samples):
            self.resample_weights()

            data_likelihood += self.trajectory_log_likelihood(observations=observations, actions=actions, initial_belief=[initial_state], num_particles=num_particles)
            fisher += self.fisher_objective(observations, actions, initial_state, num_particles, num_smoothing_samples)
            kl += self.KL_Q_to_prior()

        fisher /= num_weight_samples
        data_likelihood /= num_weight_samples
        kl /= num_weight_samples

        # we backprop on fisher
        # we want to visualize data_likelihood, but not backprop on it
        elbo = data_likelihood.detach() + fisher - fisher.detach() - kl

        return {'elbo': elbo, 'data_likelihood': data_likelihood.detach(), 'kl': kl.detach(), 'fisher': fisher.detach()}



class PoE_POMDP(POMDP):
    """POMDP where p(s'|s,a) and p(o|s) are products of experts over weighted rules.

    dynamics_rules:        list of Expert subclasses with a .transition(state, action) method
    dynamics_distributions: list of (prior, posterior) pairs, one per dynamics rule
    render_rules:          list of Expert subclasses with a .render(state, canvas) method
    render_distributions:  list of (prior, posterior) pairs, one per render rule

    self.weights[:n_dynamics] are the dynamics weights; self.weights[n_dynamics:] are the render weights.
    """

    def __init__(self, dynamics_rules, dynamics_distributions, render_rules, render_distributions, bottomup_proposal=None,width=4, height=4):
        super().__init__(list(dynamics_distributions) + list(render_distributions))
        self.dynamics_rules = list(dynamics_rules)
        self.render_rules = list(render_rules)
        self.width = width
        self.height = height

        self.bottomup_proposal = bottomup_proposal

    @property
    def dynamics_weights(self):
        return self.weights[:len(self.dynamics_rules)]

    @property
    def render_weights(self):
        return self.weights[len(self.dynamics_rules):]
    
    def propose_next_state(self, prev_state, prev_action, curr_observation):
        """Propose a next state given the previous state, action, and current observation.
        Used for particle inference; can be overridden to incorporate the current observation into the proposal.
        Returns (sample, log_prob of sample)"""
        if self.bottomup_proposal is not None:
            return self.bottomup_proposal.propose_state(prev_state, prev_action, curr_observation)
        else:
            proposal = self.sample_next_state(prev_state, prev_action)
            return proposal, None

    # --- PoE distribution builders ---

    def render_distribution(self, state):
        blank_canvas = [["black" for _ in range(self.height)] for _ in range(self.width)]
        renderings = [(w, rule.render(state, copy.deepcopy(blank_canvas)))
                      for w, rule in zip(self.render_weights, self.render_rules)]
        return self._merge_nested(blank_canvas, renderings)

    def transition_distribution(self, state, action):
        rule_new_states = [(w, rule.transition(state, action))
                           for w, rule in zip(self.dynamics_weights, self.dynamics_rules)]
        return self._merge_nested(state, rule_new_states)

    # --- POMDP abstract method implementations ---

    def sample_next_state(self, state, action):
        return self._sample_nested(self.transition_distribution(state, action))

    def sample_observation_given_state(self, state):
        return self._sample_nested(self.render_distribution(state))

    def observation_log_likelihood_given_state(self, state, observation):
        return self._log_likelihood_nested(observation, self.render_distribution(state))

    def next_state_log_likelihood(self, state, next_state, action):
        return self._log_likelihood_nested(next_state, self.transition_distribution(state, action))

    # --- particle approximation (freezes weights as Deltas) ---

    def make_particle_based_approximation(self, num_particles):
        """Like the base version but also freezes each particle's weights as Delta distributions."""
        models = []
        for _ in range(num_particles):
            model = copy.deepcopy(self)
            model.resample_weights()
            model._parameter_distributions = [
                (prior, Delta(w, legal_range=post.legal_range))
                for (prior, post), w in zip(model._parameter_distributions, model.weights)
            ]
            models.append(model)
        return models

    def __str__(self):
        all_rules = self.dynamics_rules + self.render_rules
        lines = []
        for rule, w, (_, post) in zip(all_rules, self.weights, self._parameter_distributions):
            w_val = w.item() if isinstance(w, torch.Tensor) else w
            lines.append(f"  {type(rule).__name__}: weight={w_val:.3f} ~ {post}")
        return "\n".join(lines)

    def load_str(self, s):
        """Update weight posteriors (and resample weights) from a string produced by __str__."""
        import re
        all_rules = self.dynamics_rules + self.render_rules
        rule_to_idx = {type(rule).__name__: i for i, rule in enumerate(all_rules)}
        for line in s.strip().splitlines():
            line = line.strip()
            m = re.match(r'(\w+): weight=[\d.]+ ~ (.+)', line)
            if not m:
                continue
            name, dist_str = m.group(1), m.group(2)
            if name not in rule_to_idx:
                continue
            i = rule_to_idx[name]
            prior, _ = self._parameter_distributions[i]
            bm = re.match(r'Beta\(([\d.]+),\s*([\d.]+)\)', dist_str)
            dm = re.match(r'Delta\(([\d.]+),\s*range=\(([\d.,\s]+)\)\)', dist_str)
            um = re.match(r'Uniform\(([\d.]+),\s*([\d.]+)\)', dist_str)
            if bm:
                new_post = Beta(float(bm.group(1)), float(bm.group(2)))
            elif dm:
                val = float(dm.group(1))
                lo, hi = map(float, dm.group(2).split(','))
                new_post = Delta(val, legal_range=(lo, hi))
            elif um:
                new_post = Uniform((float(um.group(1)), float(um.group(2))))
            else:
                continue
            self._parameter_distributions[i] = (prior, new_post)
            self.weights[i] = new_post.sample()

    # --- PoE utilities for nested json structures ---

    def _merge_nested(self, base, weighted_values):
        competing = [(w, v) for w, v in weighted_values if isinstance(v, Categorical)]
        if competing:
            z = sum(w for w, _ in competing)
            distribution = competing[0][1] ** (competing[0][0] / z)
            for w, d in competing[1:]:
                distribution *= d ** (w / z)
            return distribution.normalize()

        if isinstance(base, dict):
            return {key: self._merge_nested(base[key], [(w, v[key]) for w, v in weighted_values])
                    for key in base}
        elif isinstance(base, list):
            # fancy logic to handle adding new elements to the list
            # assumes that new elements are always appended
            max_len = max([len(base)] + [len(v) for _, v in weighted_values if isinstance(v, list)], default=0)
            return [self._merge_nested(
                        base[i] if i < len(base) else None,
                        [(w, v[i]) for w, v in weighted_values if isinstance(v, list) and i < len(v)])
                    for i in range(max_len)]
        elif isinstance(base, tuple):
            return tuple(self._merge_nested(base[i], [(w, v[i]) for w, v in weighted_values])
                         for i in range(len(base)))
        else:
            return base

    def _sample_nested(self, structure):
        if isinstance(structure, Categorical):
            return structure.sample()
        elif isinstance(structure, dict):
            return {k: self._sample_nested(v) for k, v in structure.items()}
        elif isinstance(structure, list):
            return [self._sample_nested(v) for v in structure]
        elif isinstance(structure, tuple):
            return tuple(self._sample_nested(v) for v in structure)
        else:
            return structure

    def _log_likelihood_nested(self, data, distribution):
        if isinstance(distribution, Categorical):
            return distribution.logits.get(data, float('-inf'))
        elif isinstance(distribution, dict):
            return sum(self._log_likelihood_nested(data[k], distribution[k]) for k in data)
        elif isinstance(distribution, list):
            return sum(self._log_likelihood_nested(d, distribution[i]) for i, d in enumerate(data))
        elif isinstance(distribution, tuple):
            return sum(self._log_likelihood_nested(d, distribution[i]) for i, d in enumerate(data))
        else:
            return 0
