import torch
import math
import numbers

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
            support = set(self.logits.keys()) & set(other.logits.keys())
            return Categorical({k: self.logits[k] + other.logits[k] for k in support})
        elif isinstance(other, torch.Tensor):
            logOther = torch.log(other)
            return Categorical({k: logit + logOther for k, logit in self.logits.items()})
        else:
            raise ValueError("Unsupported type for multiplication")
        
    def __add__(self, other):
        # logsumexp the logits
        if isinstance(other, numbers.Number):
            return Categorical({k: torch.logsumexp(torch.tensor([logit, math.log(other)]), dim=0) for k, logit in self.logits.items()})
        elif isinstance(other, Categorical):
            all_keys = set(self.logits.keys()) | set(other.logits.keys())
            return Categorical({k: torch.logsumexp(torch.tensor([self.logits.get(k, float('-inf')), other.logits.get(k, float('-inf'))]), dim=0) for k in all_keys})
        elif isinstance(other, torch.Tensor):
            logOther = torch.log(other)
            return Categorical({k: torch.logsumexp(torch.tensor([logit, logOther]), dim=0) for k, logit in self.logits.items()})
        else:
            raise ValueError("Unsupported type for addition")

    def __pow__(self, exponent):
        return Categorical({k: logit * exponent for k, logit in self.logits.items()})

    def sample(self):
        outcomes = list(self.logits.keys())
        log_probs = torch.tensor([self.logits[outcome] for outcome in outcomes], dtype=torch.float32)
        probs = torch.exp(log_probs)
        return outcomes[torch.multinomial(probs, num_samples=1).item()]

    def __str__(self):
        norm = self.normalize()
        entries = ", ".join(f"{k}: {math.exp(v):.3f}" for k, v in norm.logits.items())
        return f"Categorical({{{entries}}})"

    def __repr__(self):
        return self.__str__()
    
    def kl(self, other):
        if not isinstance(other, Categorical):
            raise ValueError("KL divergence can only be computed between two Categorical distributions")
        assert set(self.logits.keys()) == set(other.logits.keys()), "Both distributions must have the same support for KL divergence"
        kl_div = 0.0
        for k in self.logits.keys():
            p = math.e ** self.logits[k]
            log_p = self.logits[k]
            log_q = other.logits[k]
            kl_div += p * (log_p - log_q)
        return kl_div
    
    def entropy(self):
        norm = self.normalize()
        return -sum((math.e**logit) * logit for logit in norm.logits.values())
    


class Uniform:

    def __init__(self, legal_range=(0.0, 1.0)):
        self.legal_range = legal_range
        
    
    def log_prob(self, value):
        return torch.tensor(1./(self.legal_range[1] - self.legal_range[0])).log()

    def __str__(self):
        return f"Uniform({self.legal_range[0]}, {self.legal_range[1]})"

    def __repr__(self):
        return self.__str__()

class Delta:

    def __init__(self, value, legal_range=(0.0, 1.0)):
        assert legal_range[0] < value < legal_range[1], "Value must be within the legal range of the distribution"
        self.legal_range = legal_range
        # reparameterize so that we can optimize a raw value over the entire reels, which gets squashed into the legal range by a sigmoid
        # output=min+sigmoid(raw)*(max-min)
        raw_transformed_value = torch.log(torch.tensor((value - legal_range[0]) / (legal_range[1] - value)))
        self._raw = raw_transformed_value.requires_grad_(True)

    @property
    def value(self):
        return self.legal_range[0] + torch.sigmoid(self._raw) * (self.legal_range[1] - self.legal_range[0])

    def sample(self):
        return self.value

    def parameters(self):
        return (self._raw,)

    def entropy(self):
        return 0.0
    
    def log_prob(self, value):
        print("Warning: log_prob is not well-defined for a Delta distribution; returning 0")
        return 0

    def __str__(self):
        return f"Delta({self.value.item():.4f}, range={self.legal_range})"

    def __repr__(self):
        return self.__str__()

class Beta:

    def __init__(self, alpha, beta, epsilon=0.01):
        self.legal_range = (0.0, 1.0)
        self.epsilon = epsilon
        a = torch.as_tensor(alpha, dtype=torch.float32)
        b = torch.as_tensor(beta, dtype=torch.float32)
        # Invert alpha = softplus(log_alpha) + epsilon → log_alpha = log(expm1(alpha - epsilon))
        self.log_alpha = torch.log(torch.expm1(a - epsilon)).requires_grad_(True)
        self.log_beta = torch.log(torch.expm1(b - epsilon)).requires_grad_(True)

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.log_alpha) + self.epsilon

    @property
    def beta(self):
        return torch.nn.functional.softplus(self.log_beta) + self.epsilon

    def sample(self):
        return torch.distributions.Beta(self.alpha, self.beta).rsample()

    def parameters(self):
        return self.log_alpha, self.log_beta
    
    def entropy(self):
        return torch.distributions.Beta(self.alpha, self.beta).entropy()

    def log_prob(self, value):
        return torch.distributions.Beta(self.alpha, self.beta).log_prob(value)

    def __str__(self):
        return f"Beta({self.alpha.item():.4f}, {self.beta.item():.4f})"

    def __repr__(self):
        return self.__str__()