"""
# model:

weights ~ P_prior
P(state_{t+1} | state_t, action_t, weights) = product of experts (using *dynamics rules*)
P(observation_t | state_t, weights) = product of experts (using *render rules*)

This model implicitly defines:
P(observations | actions, weights), by marginalizing out states
P(states | actions, observations, weights), which is approximated with particle smoothing. Expensive!

## implementation details

Note that I defined P(state_{t+1} | state_t, action_t, weights) as a PoE, but could be many things! Just has to be differentiable w.r.t. weights
This is hacky, and its hacky-ness comes from merging different predictions of the next state from different rules.
The state is essentially json, and each rule predicts a distribution over edits to the json
When multiple rules try to edit the same part of the json, we form a PoE using their weights.
Rules can overwrite json list elements, and append to lists
Deletion not implemented because you can hack it by overwriting with None


# variational posterior: q_theta(weights)

maximize ELBO: E_{weights ~ q_theta} [ log P(observations | actions, weights) ] - KL(q_theta(weights) || P_prior(weights))

So need:
grad_theta E_{weights ~ q_theta} [ log P(observations | actions, weights) ]

Reparameterization for lets us push the gradient inside the expectation, but we still need:
grad_weights log P(observations | actions, weights)

Which is
E_{states ~ P(states | actions, observations, weights)} [ grad_weights log P(observations, states | actions, weights) ]

P(states | actions, observations, weights) is approximated with particle smoothing.

# what we expect to see from the model

With a sufficiently expressive variational family, we should see that it encodes more uncertainty about rules that are seldom used
To test this there is a DummyDynamics rule that never applies, so the model should learn to be uncertain about it (if the variational family can express it)

To handle occlusion, different rules can simultaneously "write" to the same pixel, and the weights should determine which gets priority
So we expect that the rendering rule for food has higher weight compared to the rendering rule for bugs
...But there is only one observation where there is occlusion, so we also expect there to be remaining uncertainty about the rendering weights

When rules conflict, the overriding rule gets a higher weight:
Food is stationary by default, but this is overridden by it being eaten, so want FoodDisappearsDynamics.weight > FoodStationaryDynamics.weight

When a rule always predicts correctly, but there's no alternative rule, then its weight should also default to the prior:
We have no reason to say how strong it is, because there's been no comparison rule.
If we think this is bad we could introduce a default dummy "frame axiom" rule that just says "everything stays the same"

We expect training to be super slow, because it literally has a particle filter in the inner loop.
Also, the particle filter has to work, and it's the most basic naive approach, which for ants requires a tiny grid.

Also apparently beta's are known for numerical instability, so we get nan's if we train long enough :)
"""
import copy

import random
import matplotlib.pyplot as plt
import math
import numbers
import torch
from tqdm import tqdm
from distributions import *

random.seed(0)
torch.manual_seed(0)



class Expert:

    def __init__(self, weight_posterior, weight_prior=None):
        self.weight_posterior = weight_posterior
        self.weight_prior = weight_prior if weight_prior is not None else Uniform(self.weight_posterior.legal_range)
        self.weight = self.weight_posterior.sample()

    def parameters(self):
        return self.weight_posterior.parameters() + self.weight_prior.parameters()
    
    def resample_weight(self):
        self.weight = self.weight_posterior.sample()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(result, k, v.detach().clone())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    

class RandomWalkDynamics(Expert):

    def transition(self, state, action):
        new_state = copy.deepcopy(state)
        for i, (x, y) in enumerate(state["bugs"]):
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y)]
            neighbor_distribution = Categorical({(x,y): 10.*( (x,y) in neighbors) for x in range(W) for y in range(W)}).normalize()
            new_state["bugs"][i] = neighbor_distribution
        return new_state

class ChaseDynamics(Expert):

    def transition(self, state, action):
        new_state = copy.deepcopy(state)
        for i, (x, y) in enumerate(state["bugs"]):
            food = [f for f in state["food"] if f is not None]
            if any(food):
                # Move each bug toward the food that it's closest to
                closest_food = min(food, key=lambda f: (f[0] - x) ** 2 + (f[1] - y) ** 2)
                dx = closest_food[0] - x
                dy = closest_food[1] - y
                if dx == 0 and dy == 0:
                    new_position = (x, y)  # already on food, stay put
                elif abs(dx) > abs(dy):
                    new_position = (x + (1 if dx > 0 else -1), y)
                else:
                    new_position = (x, y + (1 if dy > 0 else -1))
            else:
                # if no food, stay put
                new_position = (x, y)
            # smear probability everywhere but put most on the new position
            new_state["bugs"][i] = Categorical({(new_x, new_y): 20.0*(new_x==new_position[0] and new_y==new_position[1])
                                                for new_x in range(W) for new_y in range(H)}).normalize()
        return new_state
    
class FoodAppearsDynamics(Expert):

    def transition(self, state, action):
        global W, H

        new_state = copy.deepcopy(state)
        if action == "click":
            food = [f for f in state["food"] if f is not None]
            if food: return new_state  # if there's already food, do nothing

            # add two foods at random locations, categorical over all pairs of locations
            possible_locations = [(x, y) for x in range(W) for y in range(H)]
            dist = Categorical({loc: 0. for loc in possible_locations}).normalize()

            new_state["food"] = state["food"] + [dist, dist]
        
        return new_state
    
class FoodStationaryDynamics(Expert):

    def transition(self, state, action):
        """food doesn't run away!"""
        new_state = copy.deepcopy(state)
        for i, food in enumerate(state["food"]):
            if food is not None:
                x, y = food
                new_state["food"][i] = Categorical({(x, y): 10.0, None: 0.0}).normalize()
        return new_state
    
class DummyDynamics(Expert):
    """The point of this expert is to be untestable, so that we can verify that the model learns to be uncertain about it"""

    def transition(self, state, action):
        if action == "double click": # this never happens
            assert 0
        return copy.deepcopy(state)
    
class ArrowKeyDynamics(Expert):
    """This expert predicts that the bugs can be moved with the arrow keys, but this never happens in the data, so we expect the model to learn to be uncertain about it but to test it during active learning"""

    def transition(self, state, action):
        new_state = copy.deepcopy(state)
        if action == "up":
            new_state["bugs"] = [ Categorical({(new_x, new_y): 10.0*(new_x==x and new_y==y-1) for new_x in range(W) for new_y in range(H)}).normalize() 
                                 for x, y in state["bugs"] ]
        elif action == "down":
            new_state["bugs"] = [ Categorical({(new_x, new_y): 10.0*(new_x==x and new_y==y+1) for new_x in range(W) for new_y in range(H)}).normalize() 
                                 for x, y in state["bugs"] ]
        elif action == "left":
            new_state["bugs"] = [ Categorical({(new_x, new_y): 10.0*(new_x==x-1 and new_y==y) for new_x in range(W) for new_y in range(H)}).normalize() 
                                 for x, y in state["bugs"] ] 
        elif action == "right":
            new_state["bugs"] = [ Categorical({(new_x, new_y): 10.0*(new_x==x+1 and new_y==y) for new_x in range(W) for new_y in range(H)}).normalize() 
                                 for x, y in state["bugs"] ]
        return new_state
        
    
class FoodDisappearsDynamics(Expert):

    def transition(self, state, action):
        new_state = copy.deepcopy(state)   
        for i in range(len(state["food"])):
            if any(bug == state["food"][i] for bug in state["bugs"]):
                # Deletion means going to None
                new_state["food"][i] = Categorical({state["food"][i]: 0., None: 10.}).normalize()

        return new_state

class RenderBugRule(Expert):

    def render(self, state, observation):
        for bug in state["bugs"]:
            x, y = bug
            if 0 <= x < len(observation) and 0 <= y < len(observation[0]):
                # Note that to handle occlusion probabilistically, we have to put nonzero probability on all the other colours
                # This probability can be essentially infinitesimal
                observation_distribution = Categorical({"grey": 5, "black": -5, "green": -5}).normalize()
                observation[x][y] = observation_distribution
        
        return observation
    
class RenderFoodRule(Expert):

    def render(self, state, observation):
        for x, y in [ f for f in state["food"] if f is not None ]:
            if 0 <= x < len(observation) and 0 <= y < len(observation[0]):
                observation_distribution = Categorical({"green": 5, "black": -5, "grey": -5}).normalize()
                observation[x][y] = observation_distribution
        
        return observation

class WorldModel:

    def __init__(self, dynamics_rules, render_rules, width = 4, height = 4):
        self.dynamics_rules = dynamics_rules
        self.render_rules = render_rules
        self.width = width
        self.height = height

    def render_distribution(self, state):
        blank_canvas = [ ["black" for _ in range(self.height)] for _ in range(self.width)]
        renderings = [ (rule.weight, rule.render(state, copy.deepcopy(blank_canvas))) for rule in self.render_rules ]
        return self._merge_nested(blank_canvas, renderings)
    
    def transition_distribution(self, state, action):
        rule_new_states = [ (rule.weight, rule.transition(state, action)) for rule in self.dynamics_rules ]
        return self._merge_nested(state, rule_new_states)
    
    def parameters(self):
        return [p for rule in self.dynamics_rules + self.render_rules
                for p in rule.weight_posterior.parameters()]
    

    """hacky stuff below for defining PoE for json, doesn't have to be this way"""
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
        
    def render_log_likelihood(self, state, observation):
        render_dist = self.render_distribution(state)
        return self._log_likelihood_nested(observation, render_dist)

    def rollout(self, initial_state, actions):
        state = copy.deepcopy(initial_state)
        states = [state]
        observations = [self._sample_nested(self.render_distribution(state))]
        for action in actions:
            state = self._sample_nested(self.transition_distribution(state, action))
            states.append(state)
            observations.append(self._sample_nested(self.render_distribution(state)))
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
                new_particle = self._sample_nested(self.transition_distribution(particle, action))
                particle_weight = self.render_log_likelihood(new_particle, obs)
                particle_weights.append(particle_weight)
                new_belief.append((particle_weight, new_particle))
            # Normalize weights->build distribution
            resampling_distribution = Categorical({i: w for i, (w, _) in enumerate(new_belief)}).normalize()
            #print([math.exp(w) for w in resampling_distribution.logits.values()])
            # Resample
            new_belief = [new_belief[resampling_distribution.sample()][1] for _ in range(num_particles)]
            belief_states.append(new_belief)

            # update logZ
            logZ += torch.logsumexp(torch.tensor(particle_weights), dim=0) - math.log(num_particles)
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
                    weights.append(self._log_likelihood_nested(next_state, self.transition_distribution(particle, actions[t])))
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
            objective += sum(self.render_log_likelihood(state, obs) for state, obs in zip(trajectory, observations))
            # iterate adjacent state pairs in trajectory and actions, and add log p(s_{t+1} | s_t, a_t) to objective
            for t in range(len(trajectory) - 1):
                objective += self._log_likelihood_nested(trajectory[t + 1], self.transition_distribution(trajectory[t], actions[t]))
            num_state_samples += 1
            if num_state_samples >= num_smoothing_samples:
                break

        return objective / num_state_samples
    
    def KL_Q_to_prior(self, n_samples=50):
        """KL(q(weights) || p(weights)), in expectation"""
        kl = 0
        for _ in range(n_samples):
            for rule in self.dynamics_rules + self.render_rules:
                w = rule.weight_posterior.sample() # sample q
                kl += rule.weight_posterior.log_prob(w) - rule.weight_prior.log_prob(w)
        return kl/n_samples
    
    def elbo(self, observations, actions, initial_state, num_particles, num_weight_samples, num_smoothing_samples=1):

        fisher = 0
        data_likelihood = 0
        kl = 0
        for _ in range(num_weight_samples):
            self.resample_weights()

            data_likelihood += self.particle_filter(initial_belief=[initial_state], observations=observations, actions=actions, num_particles=num_particles)[1]
            fisher += self.fisher_objective(observations, actions, initial_state, num_particles, num_smoothing_samples)
            kl += self.KL_Q_to_prior()

        fisher /= num_weight_samples
        data_likelihood /= num_weight_samples
        kl /= num_weight_samples

        # we backprop on fisher
        # we want to visualize data_likelihood, but not backprop on it
        elbo = data_likelihood.detach() + fisher - fisher.detach() - kl

        return {'elbo': elbo, 'data_likelihood': data_likelihood.detach(), 'kl': kl.detach(), 'fisher': fisher.detach()}
    
    def train(self, observations, actions, initial_state, num_particles=1000, num_weight_samples=1, num_smoothing_samples=1, lr=0.01, iterations=100, dump_every=None):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elbo_curve = []
        fisher_curve = []
        kl_curve = []
        data_likelihood_curve = []
        pbar = tqdm(range(iterations))
        for iteration in pbar:
            optimizer.zero_grad()
            elbo = self.elbo(observations, actions, initial_state, num_particles, num_weight_samples, num_smoothing_samples)
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
    
    def resample_weights(self):
        for rule in self.dynamics_rules + self.render_rules:
            rule.resample_weight()

    def __str__(self):
        lines = []
        for rule in self.dynamics_rules + self.render_rules:
            w = rule.weight.item() if isinstance(rule.weight, torch.Tensor) else rule.weight
            lines.append(f"  {type(rule).__name__}: weight={w:.3f} ~ {rule.weight_posterior}")
        return "\n".join(lines)

    def load_str(self, s):
        """Update weight posteriors (and resample weights) from a string produced by __str__."""
        import re
        rule_map = {type(rule).__name__: rule for rule in self.dynamics_rules + self.render_rules}
        for line in s.strip().splitlines():
            line = line.strip()
            m = re.match(r'(\w+): weight=[\d.]+ ~ (.+)', line)
            if not m:
                continue
            name, dist_str = m.group(1), m.group(2)
            if name not in rule_map:
                continue
            rule = rule_map[name]
            bm = re.match(r'Beta\(([\d.]+),\s*([\d.]+)\)', dist_str)
            dm = re.match(r'Delta\(([\d.]+),\s*range=\(([\d.,\s]+)\)\)', dist_str)
            um = re.match(r'Uniform\(([\d.]+),\s*([\d.]+)\)', dist_str)
            if bm:
                rule.weight_posterior = Beta(float(bm.group(1)), float(bm.group(2)))
            elif dm:
                val = float(dm.group(1))
                lo, hi = map(float, dm.group(2).split(','))
                rule.weight_posterior = Delta(val, legal_range=(lo, hi))
            elif um:
                rule.weight_posterior = Uniform((float(um.group(1)), float(um.group(2))))
            rule.resample_weight()

    def make_particle_based_approximation(self, num_particles):
        """Samples from the posterior over *world models*, returning a list of WorldModel instances with delta-function weights."""
        models = []
        for _ in range(num_particles):
            model = copy.deepcopy(self)
            model.resample_weights()
            for rule in model.dynamics_rules + model.render_rules:
                rule.weight_posterior = Delta(rule.weight, legal_range=rule.weight_posterior.legal_range)
            models.append(model)
        return models
    
    def plan_info_gain(self, policy, initial_belief, observations, actions, num_rollouts=1, num_model_particles=3, num_state_particles=1000):
        """
        Monte Carlo estimate of expected information gain from following the provided policy, starting from belief_state
        Policy is a function that takes in a history of (observations, actions) and returns an action, or None to indicate the end of the episode
        Returns a list of info gains at each time step

        observatoins: o1, o2, ..., oT
        actions: a1, a2, ..., aT-1
        initial_belief: p(state_1) (note that this is a belief over the first state, not the first observation, so we ignore o1)
        """

        assert len(observations) == len(actions) + 1, "There should be one more observation than actions, since the first observation is before any action is taken"

        # belief state is a tuple of (world model, latent state)
        belief_state = []

        # information gain computed with respect to this distribution over world models
        possible_models = self.make_particle_based_approximation(num_model_particles)
        # for each possible model, run a particle filter to get possible they tend the states
        for model in tqdm(possible_models, desc="Running particle filters for each possible model"):
            for latent_state in model.particle_filter(initial_belief=initial_belief, observations=observations, actions=actions, num_particles=num_state_particles)[0][-1]:
                belief_state.append((model, latent_state))


        # produce rollout
        root_belief_state, root_actions, root_observations = belief_state, actions, observations
        info_gains = [list() for _ in range(10)]  # hacky way to store info gains at each time step, assuming no more than 10 time steps
        for _ in range(num_rollouts):
            belief_state, actions, observations = root_belief_state, root_actions, root_observations
            
            while True:
                a = policy(observations, actions)
                if a is None: break
                actions = actions + [a]
                
                true_model, true_state = random.choice(belief_state)
                true_next_state =true_model._sample_nested(true_model.transition_distribution(true_state, a))
                observations = observations + [true_model._sample_nested(true_model.render_distribution(true_next_state))]

                print("TRANSITIONED:")
                print(true_state)
                # pretty print the observation
                print_observation(observations[-2])
                print("ACTION TAKEN:", a)
                print("NEXT STATE:")
                print(true_next_state)
                print_observation(observations[-1])
                print("\n\n")


                # update belief state
                log_weights = []
                new_beliefs = []
                for (model, latent_state) in belief_state:
                    new_latent_state = model._sample_nested(model.transition_distribution(latent_state, a))
                    log_weight = model._log_likelihood_nested(observations[-1], model.render_distribution(new_latent_state))
                    log_weights.append(log_weight)
                    new_beliefs.append((model, new_latent_state))

                # Normalize weights
                weight_dist = Categorical({i: w for i, w in enumerate(log_weights)}).normalize()

                new_belief_state = []
                for _ in belief_state:
                    i = weight_dist.sample()
                    new_belief_state.append((new_beliefs[i][0], new_beliefs[i][1]))

                # compute KL between old and new marginal over models
                for i, model in enumerate(possible_models):
                    print(f"Model {i} weight: {sum(1 for b in belief_state if id(b[0]) == id(model))} -> {sum(1 for b in new_belief_state if id(b[0]) == id(model))}")
                old_model_weights = Categorical({id(model): math.log(1e-10+sum( id(model) == id(b[0]) for b in belief_state)) for model in possible_models}).normalize()
                new_model_weights = Categorical({id(model): math.log(1e-10+sum( id(model) == id(b[0]) for b in new_belief_state)) for model in possible_models}).normalize()
                print(f"entropy change: {old_model_weights.entropy():.4f} -> {new_model_weights.entropy():.4f}")
                infogain = old_model_weights.entropy() - new_model_weights.entropy()

                # show the model that most changed in weight
                model_weight_changes = {id(model): sum( id(model) == id(b[0]) for b in new_belief_state) - sum( id(model) == id(b[0]) for b in belief_state) for model in possible_models}
                most_changed_model_id = max(model_weight_changes, key=lambda mid: abs(model_weight_changes[mid]))
                for model in possible_models:
                    if id(model) == most_changed_model_id:
                        print("Most changed model:")
                        print(model)
                        break

                timestep = len(actions) - len(root_actions) - 1
                if len(info_gains) <= timestep: info_gains.append(list())
                info_gains[timestep].append(infogain)

                belief_state = new_belief_state

        return [sum(gains)/len(gains) if gains else 0. for gains in info_gains]
        

            
            
        
    

                

        

W, H, T = 4, 4, 4
initial_state = {
    "bugs": [(0, 0), (2, 3)],
    "food": []}

"""Create a world model where the chase dynamics dominates the random walk dynamics, and the food is rendered over the bugs"""
actual_model = WorldModel(
    dynamics_rules=[RandomWalkDynamics(weight_posterior=Delta(1., legal_range=(0.01, 10.))),
                    ChaseDynamics(weight_posterior=Delta(9.0, legal_range=(0.01, 10.))),
                    FoodStationaryDynamics(weight_posterior=Delta(1.0, legal_range=(0.01, 10.))),
                    FoodAppearsDynamics(weight_posterior=Delta(9.0, legal_range=(0.01, 10.))),
                    FoodDisappearsDynamics(weight_posterior=Delta(9.0, legal_range=(0.01, 10.)))],
    render_rules=[RenderBugRule(weight_posterior=Delta(1.0, legal_range=(0.01, 10.))),
                  RenderFoodRule(weight_posterior=Delta(9.0, legal_range=(0.01, 10.)))],
    width=W, height=H)
actions = ["nothing"] + ["click"] * (T - 1)
print(actual_model.parameters())
states, observations = actual_model.rollout(initial_state, actions=actions)

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

def print_observation(observation):
    # transpose observation
    observation = list(zip(*observation))
    for row in observation:
        print("".join(render_cell(cell) for cell in row))

print("Training data (states will be hidden):")
for t, (state, observation) in enumerate(zip(states, observations)):
    print(f"Time step {t}:")
    for bug in state["bugs"]:
        print(f"  Bug at position {bug}")
    for food in state["food"]:
        print(f"  Food at position {food}")
    print_observation(observation)
    print()

logZ = actual_model.particle_filter(initial_belief=[initial_state], observations=observations, actions=actions, num_particles=1000)[1]
print(f"Log marginal likelihood of data under actual model: {logZ.item():.2f}")

def run_experiment(posterior_factory, prior_factory, label, actual_logZ=None, steps=1000):
    """
    posterior_factory: callable() -> fresh weight posterior per rule
    prior_factory:     callable() -> fresh weight prior per rule, or None to use default (Uniform over posterior's legal_range)
    """
    print(f"\n{'='*60}\nExperiment: {label}\n{'='*60}")

    def make_rule(cls):
        post = posterior_factory()
        prior = prior_factory() if prior_factory is not None else None
        return cls(weight_posterior=post, weight_prior=prior) if prior is not None else cls(weight_posterior=post)

    learned_model = WorldModel(
        dynamics_rules=[make_rule(RandomWalkDynamics),
                        make_rule(ChaseDynamics),
                        make_rule(DummyDynamics),
                        make_rule(ArrowKeyDynamics),
                        make_rule(FoodStationaryDynamics),
                        make_rule(FoodAppearsDynamics),
                        make_rule(FoodDisappearsDynamics)],
        render_rules=[make_rule(RenderBugRule),
                      make_rule(RenderFoodRule)],
        width=W, height=H)

    slug = label.replace(" ", "_").replace("(", "").replace(")", "")

    def dump_plots(step, training_curves):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (key, title) in zip(axes, [('elbo_curve', 'ELBO'), ('data_likelihood_curve', 'Data log-likelihood'),
                                            ('kl_curve', 'KL penalty'), ('fisher_curve', 'Fisher objective')]):
            ax.plot(training_curves[key])
            if key == 'data_likelihood_curve' and actual_logZ is not None:
                ax.axhline(actual_logZ, color='red', linestyle='--', label='actual model')
                ax.legend(fontsize=8)
            ax.set_title(title)
        fig.suptitle(f"{label} — training curves (step {step})")
        plt.tight_layout()
        fig.savefig(f"export/{slug}_training_curves_step{step:05d}.png")
        # plt.show()

        all_rules = learned_model.dynamics_rules + learned_model.render_rules
        fig, axes = plt.subplots(1, len(all_rules), figsize=(4 * len(all_rules), 4))
        for ax, rule in zip(axes, all_rules):
            posterior = rule.weight_posterior
            prior = rule.weight_prior
            lo, hi = posterior.legal_range

            samples = [posterior.sample().item() for _ in range(1000)]
            ax.hist(samples, bins=20, range=(lo, hi), density=True, alpha=0.6, label="posterior")

            xs = torch.linspace(lo, hi, 200)
            if isinstance(prior, Uniform):
                density = 1.0 / (hi - lo)
                ax.axhline(density, color="orange", label="prior")
            elif isinstance(prior, Beta):
                with torch.no_grad():
                    ys = torch.exp(prior.log_prob(xs)).numpy()
                ax.plot(xs.numpy(), ys, color="orange", label="prior")

            ax.set_xlim(lo, hi)
            ax.set_title(type(rule).__name__)
            ax.legend(fontsize=7)

        fig.suptitle(f"{label} — weight posteriors vs priors (step {step})")
        plt.tight_layout()
        fig.savefig(f"export/{slug}_posteriors_step{step:05d}.png")
        # plt.show()

    logZ_before = learned_model.particle_filter(initial_belief=[initial_state], observations=observations, actions=actions, num_particles=500)[1]
    print(f"Log marginal likelihood before training: {logZ_before.item():.2f}")

    for step, training_curves in learned_model.train(observations, actions, initial_state, num_particles=500, num_weight_samples=1, num_smoothing_samples=1, lr=0.01, iterations=steps, dump_every=50):
        dump_plots(step, training_curves)

    logZ_after = learned_model.particle_filter(initial_belief=[initial_state], observations=observations, actions=actions, num_particles=500)[1]
    print(f"Log marginal likelihood after training:  {logZ_after.item():.2f}")

    for rule in learned_model.dynamics_rules + learned_model.render_rules:
        print(f"  {type(rule).__name__}: weight={rule.weight.item():.3f} ~ {rule.weight_posterior}")

    return learned_model

def do_nothing_policy(observations, actions):
    if len(actions) < 1:
        return "nothing"
    return None

def arrow_key_policy(observations, actions):
    if len(actions) < 1: 
        return "up"
    return None

def click_policy(observations, actions):
    if len(actions) < 1:
        return "click"
    return None

model = run_experiment(
    posterior_factory=lambda: Beta(1.0, 1.0),
    prior_factory=lambda: Uniform(legal_range=(0.0, 1.0)),
    label="Beta posterior (variational)",
    actual_logZ=logZ.item(),
    steps=1)

model_checkpoint = """  RandomWalkDynamics: weight=0.224 ~ Beta(0.9566, 1.0693)
  ChaseDynamics: weight=0.523 ~ Beta(1.4035, 1.0076)
  DummyDynamics: weight=0.470 ~ Beta(1.0048, 1.0197)
  ArrowKeyDynamics: weight=0.045 ~ Beta(1.0185, 0.9932)
  FoodStationaryDynamics: weight=0.276 ~ Beta(0.8144, 1.3690)
  FoodAppearsDynamics: weight=0.766 ~ Beta(0.9985, 1.0192)
  FoodDisappearsDynamics: weight=0.905 ~ Beta(1.5161, 0.8621)
  RenderBugRule: weight=0.050 ~ Beta(0.7586, 1.6388)
  RenderFoodRule: weight=0.631 ~ Beta(1.7600, 0.6733)"""
model.load_str(model_checkpoint)

# now we try active learning
plt.figure(figsize=(8, 4))
for policy, label in [(do_nothing_policy, "do-nothing policy"), (arrow_key_policy, "arrow-key policy"), (click_policy, "click policy")]:
    info_gains = model.plan_info_gain(policy, initial_belief=[initial_state], observations=observations[:1], actions=[], num_model_particles=20, num_state_particles=500, num_rollouts=10)
    plt.plot(info_gains, label=label)
    print(f"Expected information gain from {label}: {info_gains} -> {sum(info_gains):.4f} nats")
plt.title("Expected information gain from different policies")
plt.xlabel("Time step")
plt.ylabel("Expected information gain (nats)")
plt.legend()
plt.savefig("export/expected_information_gain.png")

run_experiment(
    posterior_factory=lambda: Delta(1.0, legal_range=(0.01, 10.)),
    prior_factory=None,
    label="Delta posterior (point estimate)",
    actual_logZ=logZ.item(),
    steps=100)

