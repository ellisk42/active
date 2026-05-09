
import matplotlib.pyplot as plt
import math
import numbers
import torch

class Categorical:

    def __init__(self, logits):
        self.logits = logits

    @property
    def Z(self):
        vals = list(self.logits.values())
        if any(isinstance(v, torch.Tensor) for v in vals):
            stacked = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(float(v)) for v in vals])
            return torch.logsumexp(stacked, dim=0)
        return math.log(sum(math.exp(v) for v in vals))

    def normalize(self):
        Z = self.Z
        return Categorical({k: logit - Z for k, logit in self.logits.items()})

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Categorical({k: logit + math.log(other) for k, logit in self.logits.items()})
        elif isinstance(other, Categorical):
            return Categorical({k: self.logits[k] + other.logits[k] for k in self.logits})
        else:
            raise ValueError("Unsupported type for multiplication")

    def __pow__(self, exponent):
        return Categorical({k: logit * exponent for k, logit in self.logits.items()})

    def sample(self):
        outcomes = list(self.logits.keys())
        log_probs = torch.tensor([self.logits[outcome] for outcome in outcomes], dtype=torch.float32)
        probs = torch.exp(log_probs)
        return outcomes[torch.multinomial(probs, num_samples=1).item()]

class Uniform:
    """Uniform prior on (0, 1): log_prob = 0 everywhere."""
    def log_prob(self, value):
        return torch.tensor(0.0)

class Delta:

    def __init__(self, value):
        v = torch.as_tensor(value, dtype=torch.float32)
        self._raw = torch.log(v / (1 - v)).requires_grad_(True)

    @property
    def value(self):
        return torch.sigmoid(self._raw)

    def sample(self):
        return self.value

    def parameters(self):
        return (self._raw,)

    def entropy(self):
        return 0.0
    
class Beta:

    def __init__(self, alpha, beta):
        a = torch.as_tensor(alpha, dtype=torch.float32)
        b = torch.as_tensor(beta, dtype=torch.float32)
        self.log_alpha = torch.log(torch.exp(a) - 1).requires_grad_(True)
        self.log_beta = torch.log(torch.exp(b) - 1).requires_grad_(True)

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.log_alpha)

    @property
    def beta(self):
        return torch.nn.functional.softplus(self.log_beta)

    def sample(self):
        return torch.distributions.Beta(self.alpha, self.beta).rsample()

    def parameters(self):
        return self.log_alpha, self.log_beta
    
    def entropy(self):
        return torch.distributions.Beta(self.alpha, self.beta).entropy()

    def log_prob(self, value):
        return torch.distributions.Beta(self.alpha, self.beta).log_prob(value)

class Expert:

    def __init__(self, output_distribution, weight_distribution, prior_distribution=None, name=None):
        self.output_distribution = output_distribution
        self.weight_distribution = weight_distribution
        self.prior_distribution = prior_distribution if prior_distribution is not None else Beta(0.1, 0.1)
        self.name = name

    def predict(self):
        return self.output_distribution
    

class PoE:
    
    def __init__(self, experts):
        self.experts = experts

    def predict(self, visualize=False):
        prediction = None

        for expert in self.experts:
            w = expert.weight_distribution.sample()
            p = expert.predict() ** w
            if visualize:
                print(f"{expert.name}: weight={w.item():.3f}")
                print(f"{expert.name}: prediction^w[A]={p.logits['A']:.3f}, prediction^w[B]={p.logits['B']:.3f}")
            
            if prediction is None:
                prediction = p
            else:
                prediction = prediction * p
        normalized_prediction = prediction.normalize()
        if visualize:
            print(f"Combined prediction (normalized): p[A]={normalized_prediction.logits['A']:.3f}, p[B]={normalized_prediction.logits['B']:.3f}")
        return normalized_prediction

    def elbo(self, data):
        log_likelihood = sum( self.predict().logits[datum] for datum in data )
        kl = sum(
            -expert.weight_distribution.entropy()
            - sum(expert.prior_distribution.log_prob(expert.weight_distribution.sample()) for _ in range(len(data))) / len(data)
            for expert in self.experts
        )
        return log_likelihood, kl

    def fit(self, data, lr=0.01, iterations=1000):
        curves = {'elbo': [], 'log_likelihood': [], 'kl': []}
        params = [p for expert in self.experts
                  for p in expert.weight_distribution.parameters()]
        optimizer = torch.optim.Adam(params, lr=lr)
        def closure():
            optimizer.zero_grad()
            ll, kl = self.elbo(data)
            loss = -(ll - kl)
            loss.backward()
            return loss
        for _ in range(iterations):
            optimizer.step(closure)
            with torch.no_grad():
                ll, kl = self.elbo(data)
            curves['log_likelihood'].append(ll.item())
            curves['kl'].append(kl.item())
            curves['elbo'].append((ll - kl).item())
        return curves


def make_experts(use_delta=False):
    if use_delta:
        weight_dist = lambda: Delta(0.5)
        prior_dist  = lambda: Uniform()
    else:
        weight_dist = lambda: Beta(1, 1)
        prior_dist  = lambda: None  # uses Expert default Beta(0.1, 0.1)
    return [
        Expert(Categorical({'A': 0,   'B': -10 }).normalize(), weight_dist(), prior_distribution=prior_dist(), name="always A"),
        Expert(Categorical({'A': -10, 'B': 0   }).normalize(), weight_dist(), prior_distribution=prior_dist(), name="always B"),
        Expert(Categorical({'A': 0,   'B': -0.5}).normalize(), weight_dist(), prior_distribution=prior_dist(), name="mostly A"),
    ]

conditions   = [('Bayesian', False), ('Point estimate', True)]
dataset_sizes = [10, 50, 100]

fig, axes = plt.subplots(len(dataset_sizes), 5, figsize=(20, 3 * len(dataset_sizes)))
pa_values = {name: [] for name, _ in conditions}

for row, dataset_size in enumerate(dataset_sizes):
    dataset = ["A"] * dataset_size #// 2) + ["B"] * (dataset_size - dataset_size // 2)

    if row > 0:
        axes[row, 4].set_visible(False)

    for col_offset, (cond_name, use_delta) in enumerate(conditions):
        model = PoE(make_experts(use_delta))
        curves = model.fit(dataset, lr=0.01, iterations=500)

        samples = [model.predict().sample() for _ in range(100)]
        pa = samples.count("A") / len(samples)
        pa_values[cond_name].append(pa)
        print(f"{cond_name}\tn={dataset_size}\tp(A)={pa:.2f}")

        ax_post = axes[row, col_offset * 2]
        ax_loss = axes[row, col_offset * 2 + 1]

        x = torch.linspace(0.0, 1.0, 100)
        for i, expert in enumerate(model.experts):
            label = expert.name
            color = f'C{i}'
            if use_delta:
                ax_post.axvline(expert.weight_distribution.value.item(), label=label, color=color)
                ax_post.set_xlim(-0.1, 1.1)
            else:
                alpha, beta = expert.weight_distribution.alpha, expert.weight_distribution.beta
                y = torch.distributions.Beta(alpha, beta).log_prob(x).exp()
                ax_post.plot(x.detach().numpy(), y.detach().numpy(), label=label, color=color)
        ax_post.set_xlabel('Weight')
        ax_post.set_ylabel('Density')
        ax_post.set_title(f'{cond_name} posteriors (n={dataset_size})')
        ax_post.legend()

        ax_loss.plot(curves['elbo'],           label='ELBO')
        ax_loss.plot(curves['log_likelihood'], label='Log likelihood')
        ax_loss.plot(curves['kl'],             label='KL')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Nats')
        ax_loss.set_title(f'{cond_name} loss (n={dataset_size})')
        ax_loss.legend()

for cond_name, _ in conditions:
    axes[0, 4].plot(dataset_sizes, pa_values[cond_name], marker='o', label=cond_name)
axes[0, 4].set_xlabel('Dataset size')
axes[0, 4].set_ylabel('p(A)')
axes[0, 4].set_title('p(A) vs dataset size')
axes[0, 4].legend()

plt.tight_layout()
plt.show()