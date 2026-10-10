"""Create a teacher initialization with exactly the baseline's initial actions."""
import argparse
import copy
from pathlib import Path

import torch

from rma_local.env import OBS_DIM
from rma_local.models import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    initial = torch.load(args.baseline, map_location="cpu", weights_only=False)
    if initial["config"]["kind"] != "baseline":
        raise ValueError("Expected a baseline checkpoint")
    torch.manual_seed(initial["config"]["seed"])
    teacher = copy.deepcopy(initial)
    teacher["config"]["kind"] = "teacher"
    # Baseline latent inputs were always zero, so these columns never learned.
    for key in ["actor.0.weight", "critic.0.weight"]:
        teacher["policy"][key][:, OBS_DIM:] = 0.
    baseline_policy, teacher_policy = Policy("baseline"), Policy("teacher")
    baseline_policy.load_state_dict(initial["policy"])
    teacher_policy.load_state_dict(teacher["policy"])
    # Check both actor and critic against arbitrary observations and parameters.
    torch.manual_seed(13579)
    obs, privileged = torch.randn(64, OBS_DIM), torch.randn(64, 3)
    with torch.inference_mode():
        torch.testing.assert_close(baseline_policy.deterministic(obs), teacher_policy.deterministic(obs, privileged), rtol=0, atol=0)
        torch.testing.assert_close(baseline_policy.value(obs, privileged), teacher_policy.value(obs, privileged), rtol=0, atol=0)
    # Saved Adam moments for the previously unused columns must also be zero.
    parameter_ids = initial["optimizer"]["param_groups"][0]["params"]
    for (name, _), parameter_id in zip(baseline_policy.named_parameters(), parameter_ids):
        if name in ["actor.0.weight", "critic.0.weight"]:
            for key in ["exp_avg", "exp_avg_sq"]:
                if parameter_id in teacher["optimizer"]["state"]:
                    assert not teacher["optimizer"]["state"][parameter_id][key][:, OBS_DIM:].any()
    teacher["warmstart"] = dict(parent=str(Path(args.baseline).resolve()), method="copy baseline and zero previously unused actor/critic latent-input columns", inherited_control_steps=initial["total_steps"], independent_training_seed=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(teacher, out)
    print("Baseline-equivalent actor and critic verified; created", out)


if __name__ == "__main__":
    main()
