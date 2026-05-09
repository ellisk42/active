"""
code for representing experts-models that output json
defines P(new_data_struct | old_data_struct, expert_outputs)
"""

import dataclasses
class ExpertConfig(dataclasses.dataclass):
    normalize: bool # are expert weights normalized when there are competing predictions?
    mode: str # "mixture" or "product" - how to combine competing expert outputs

def sample(original_data_structure, expert_predictions, expert_config: ExpertConfig):
    """
    Sample from P(new_data_struct | old_data_struct, expert_outputs)
    """

    # leaf node
    if not isinstance(original_data_structure, (list, dict, tuple)):
        probabilistic_predictions = []
        for w, expert_prediction in expert_predictions:
            if isinstance(expert_prediction, Categorical):
                probabilistic_predictions.append((w, expert_prediction.probs))
            else:
                assert expert_prediction == original_data_structure, "Non-categorical expert prediction must match the original data structure"

        if not probabilistic_predictions:
            return original_data_structure
        
        if expert_config.normalize:
            total_weight = sum(w for w, _ in probabilistic_predictions)
            probabilistic_predictions = [(w / total_weight, p) for w, p in probabilistic_predictions]

        if expert_config.mode == "mixture":
            distribution = None
            for w, p in probabilistic_predictions:
                if distribution is None:
                    distribution = w * p
                else:
                    distribution += w * p
            return distribution.normalize().sample()
        elif expert_config.mode == "product":
            distribution = None
            for w, p in probabilistic_predictions:
                if distribution is None:
                    distribution = p ** w
                else:
                    distribution *= p ** w
            return distribution.normalize().sample()
        else:
            raise ValueError(f"Unsupported expert combination mode: {expert_config.mode}")
        
    # non-leaf node
    if isinstance(original_data_structure, list):
        