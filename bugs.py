"""
Bugs-and-food world: experts, data generation, and experiments.

# what we expect to see from the model

With a sufficiently expressive variational family, we should see that it encodes more uncertainty about rules that are seldom used.
To test this there is a DummyDynamics rule that never applies, so the model should learn to be uncertain about it (if the variational family can express it).

To handle occlusion, different rules can simultaneously "write" to the same pixel, and the weights should determine which gets priority.
So we expect that the rendering rule for food has higher weight compared to the rendering rule for bugs.
...But there is only one observation where there is occlusion, so we also expect there to be remaining uncertainty about the rendering weights.

When rules conflict, the overriding rule gets a higher weight:
Food is stationary by default, but this is overridden by it being eaten, so want FoodDisappearsDynamics.weight > FoodStationaryDynamics.weight.

When a rule always predicts correctly, but there's no alternative rule, then its weight should also default to the prior:
We have no reason to say how strong it is, because there's been no comparison rule.
If we think this is bad we could introduce a default dummy "frame axiom" rule that just says "everything stays the same".

We expect training to be super slow, because it literally has a particle filter in the inner loop.
Also, the particle filter has to work, and it's the most basic naive approach, which for ants requires a tiny grid.

Also apparently beta's are known for numerical instability, so we get nan's if we train long enough :)
"""
import copy
import pickle
import matplotlib.pyplot as plt
import torch
from model import *
import random
import numpy as np

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

W, H, T = 3, 3, 8


# --- Expert subclasses (stateless: no weights, no priors, no posteriors) ---

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
            new_state["bugs"][i] = Categorical({(new_x, new_y): 10.0*(new_x==new_position[0] and new_y==new_position[1])
                                                for new_x in range(W) for new_y in range(H)}).normalize()
        return new_state

class FoodAppearsDynamics(Expert):

    def transition(self, state, action):
        new_state = copy.deepcopy(state)
        if action == "click":
            food = [f for f in state["food"] if f is not None]
            if food: return new_state  # if there's already food, do nothing

            # add two foods at random locations, categorical over all pairs of locations
            possible_locations = [(x, y) for x in range(W) for y in range(H)]
            dist = Categorical({loc: 0. for loc in possible_locations}).normalize()

            new_state["food"] = state["food"] + [dist]

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
    """The point of this expert is to be untestable, so that we can verify that the model learns to be uncertain about it."""

    def transition(self, state, action):
        if action == "double click": # this never happens
            assert 0
        return copy.deepcopy(state)

class ArrowKeyDynamics(Expert):
    """This expert predicts that the bugs can be moved with the arrow keys, but this never happens in the data,
    so we expect the model to learn to be uncertain about it but to test it during active learning."""

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
    
class BottomUpProposalExpert(Expert):
    """A proposal expert that looks at the current observation and proposes a state that matches it."""

    def propose_state(self, prev_state, prev_action, curr_observation):
        proposed_state = {"bugs": [], "food": []}
        for x in range(len(curr_observation)):
            for y in range(len(curr_observation[0])):
                cell = curr_observation[x][y]
                if cell == "grey":
                    proposed_state["bugs"].append((x, y))
                elif cell == "green":
                    proposed_state["food"].append((x, y))
        # Note: it's possible for the bug to be "underneath" the food
        # if we didn't find the bug then assume it's under the food
        if not proposed_state["bugs"] and proposed_state["food"]:
            proposed_state["bugs"].append(proposed_state["food"][0])
        
        # now we have a silly bookkeeping problem, which is that the foods don't actually get deleted from the list and instead turn into None
        # this is an artifact of how we are managing the probabilities with the data structure and should go away with a better implementation
        # for now though we have to prepend None's
        num_nones = sum(f is None for f in prev_state["food"])
        # also maybe we got a new none, because we ate the food just now
        if len(proposed_state["food"]) == 0 and any(f is not None for f in prev_state["food"]):
            num_nones += 1
        proposed_state["food"] = [None] * num_nones + proposed_state["food"]
        return proposed_state, 0.


def print_observation(observation):
    actual_model.pretty_print_observation(observation)


# --- Shared distribution factories for the bugs domain ---

def _point(val, lo=0.01, hi=10.):
    return (Uniform((lo, hi)), Delta(val, legal_range=(lo, hi)))


# --- Ground-truth model and training data ---

initial_state = {
    "bugs": [(0, 0)],
    "food": []}

"""Create a world model where the chase dynamics dominates the random walk dynamics, and the food is rendered over the bugs"""
actual_model = PoE_POMDP(
    dynamics_rules=[RandomWalkDynamics(), ChaseDynamics(), FoodStationaryDynamics(),
                    FoodAppearsDynamics(), FoodDisappearsDynamics()],
    dynamics_distributions=[_point(1.), _point(9.), _point(1.), _point(9.), _point(9.)],
    render_rules=[RenderBugRule(), RenderFoodRule()],
    render_distributions=[_point(1.), _point(9.)],
    initial_belief=[initial_state],
    width=W, height=H, bottomup_proposal=BottomUpProposalExpert())
actions = ["nothing"] + ["click"] * 2 + ["nothing"] * (T - 3)
print(actual_model.parameters())
states, observations = actual_model.rollout(initial_state, actions=actions)

print("Training data (states will be hidden):")
for t, (state, observation, action) in enumerate(zip(states, observations, actions)):
    print(f"Time step {t}, a_{t}={action}")
    for bug in state["bugs"]:
        print(f"  Bug at position {bug}")
    for food in state["food"]:
        print(f"  Food at position {food}")
    print_observation(observation)
    print()

logZ = actual_model.particle_filter(observations=observations, actions=actions, num_particles=1)[1]
print(f"Log marginal likelihood of data under actual model: {logZ.item():.2f}")


# --- Experiment harness ---

def run_experiment(posterior_factory, prior_factory, label, actual_logZ=None, steps=1000, num_state_particles=1, num_weight_samples=1, num_trajectories=1):
    """
    posterior_factory: callable() -> fresh weight posterior per rule
    prior_factory:     callable() -> fresh weight prior per rule
    """
    print(f"\n{'='*60}\nExperiment: {label}\n{'='*60}")

    def make_dist():
        return (prior_factory(), posterior_factory())

    learned_model = PoE_POMDP(
        dynamics_rules=[RandomWalkDynamics(), ChaseDynamics(), DummyDynamics(),
                        ArrowKeyDynamics(), FoodStationaryDynamics(),
                        FoodAppearsDynamics(), FoodDisappearsDynamics()],
        dynamics_distributions=[make_dist() for _ in range(7)],
        render_rules=[RenderBugRule(), RenderFoodRule()],
        render_distributions=[make_dist() for _ in range(2)],
        initial_belief=[initial_state],
        width=W, height=H,
        bottomup_proposal=BottomUpProposalExpert())

    slug = label.replace(" ", "_").replace("(", "").replace(")", "")

    def ewa(values, alpha=0.1):
        s = values[0]
        result = []
        for v in values:
            s = alpha * v + (1 - alpha) * s
            result.append(s)
        return result

    def dump_plots(step, training_curves):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (key, title) in zip(axes, [('elbo_curve', 'ELBO'), ('data_likelihood_curve', 'Data log-likelihood'),
                                            ('kl_curve', 'KL penalty'), ('fisher_curve', 'Fisher objective')]):
            vals = training_curves[key]
            ax.plot(vals, color='steelblue', alpha=0.35, linewidth=0.8)
            ax.plot(ewa(vals), color='steelblue', linewidth=1.8, label='EWA')
            if key == 'data_likelihood_curve' and actual_logZ is not None:
                ax.axhline(actual_logZ*num_trajectories, color='red', linestyle='--', label='actual model')
            ax.legend(fontsize=8)
            ax.set_title(title)
        fig.suptitle(f"{label} — training curves (step {step})")
        plt.tight_layout()
        fig.savefig(f"export/{slug}_training_curves.png")
        plt.close(fig)

        all_rules = learned_model.dynamics_rules + learned_model.render_rules
        all_dists = learned_model.parameter_distributions()
        fig, axes = plt.subplots(1, len(all_rules), figsize=(4 * len(all_rules), 4))
        for ax, rule, (prior, posterior) in zip(axes, all_rules, all_dists):
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
        fig.savefig(f"export/{slug}_posteriors.png")
        plt.close(fig)

        with open(f"export/{slug}_checkpoint.pkl", "wb") as f:
            pickle.dump(learned_model, f)

    logZ_before = learned_model.particle_filter(observations=observations, actions=actions, num_particles=num_state_particles)[1]
    logZ_before *= num_trajectories  # we will be training on num_trajectories copies of the same trajectory, so the log-likelihood should scale linearly with this
    print(f"Log marginal likelihood before training: {logZ_before.item():.2f}")

    for step, training_curves in learned_model.train(
            lr=0.01, iterations=steps, dump_every=50,
            trajectories=[(observations, actions)]*num_trajectories,
            num_particles=num_state_particles, num_weight_samples=1, num_smoothing_samples=1):
        dump_plots(step, training_curves)

    logZ_after = learned_model.particle_filter(observations=observations, actions=actions, num_particles=num_state_particles)[1]
    logZ_after *= num_trajectories  # we will be training on num_trajectories copies of the same trajectory, so the log-likelihood should scale linearly with this
    print(f"Log marginal likelihood after training:  {logZ_after.item():.2f}")

    print(learned_model)

    with open(f"export/{slug}_final_model.pkl", "wb") as f:
        pickle.dump(learned_model, f)

    return learned_model


# --- Policies for active learning ---

def do_nothing_policy(observations, actions):
    if len(actions) < 1:
        return "nothing"
    return None

def arrow_key_policy(observations, actions):
    if len(actions) < 1:
        return "down"
    return None

def click_policy(observations, actions):
    if len(actions) < 1:
        return "click"
    return None

def arrow_click_policy(observations, actions):
    if len(actions) < 1:
        return "right"
    elif len(actions) < 2:
        return "click"
    return None


def compare_policies(learner, policies_with_labels, n_rollouts=10,
                     output_path="export/expected_information_gain.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for policy, label in policies_with_labels:
        gains = learner.rollout_policy(policy, n_rollouts=n_rollouts)
        ax.plot([0]+gains, label=label, marker='o')
        print(f"Expected information gain from {label}: {gains} -> {sum(gains):.4f} nats")
    ax.set_title("Expected information gain from different policies")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Expected information gain (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# --- Model loading ---

def load_model(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


# --- Entry point ---

if __name__ == "__main__":
    import argparse
    from active_learning import ActiveLearner

    parser = argparse.ArgumentParser(description="Bugs-and-food world model")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train via variational inference")
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--num-state-particles", type=int, default=10)
    train_p.add_argument("--num-trajectories", type=int, default=10)

    al_p = sub.add_parser("active", help="Active learning / info-gain planning")
    al_p.add_argument("--checkpoint", default="export/Beta_posterior_variational_final_model.pkl")
    al_p.add_argument("--num-model-particles", type=int, default=30)
    al_p.add_argument("--num-state-particles", type=int, default=30)
    al_p.add_argument("--num-rollouts", type=int, default=100)

    args = parser.parse_args()

    if args.command == "train":
        run_experiment(
            posterior_factory=lambda: Beta(1.0, 1.0),
            prior_factory=lambda: Beta(0.1, 0.1),
            label="Beta posterior (variational)",
            actual_logZ=logZ.item(),
            steps=args.steps,
            num_state_particles=args.num_state_particles,
            num_trajectories=args.num_trajectories)

    elif args.command == "active":
        model = load_model(args.checkpoint)
        model.num_particles = args.num_state_particles
        learner = ActiveLearner(model, history=[observations[0]], num_model_particles=args.num_model_particles)
        learner.init_tree()
        compare_policies(
            learner,
            policies_with_labels=[
                (do_nothing_policy,  "do-nothing policy"),
                (arrow_key_policy,   "arrow-key policy"),
                (click_policy,       "click policy"),
                (arrow_click_policy, "arrow-key-then-click policy"),
            ],
            n_rollouts=args.num_rollouts)
