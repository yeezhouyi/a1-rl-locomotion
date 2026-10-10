"""Export CPU TorchScript actors without critic, encoder or MuJoCo dependency."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from rma_local.env import OBS_DIM
from rma_local.models import Policy, make_adaptation


class BaselineDeployment(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        latent = observation.new_zeros((observation.shape[0], 8))
        return torch.tanh(self.actor(torch.cat([observation, latent], dim=-1)))


class RMADeployment(nn.Module):
    def __init__(self, actor, adaptation):
        super().__init__()
        self.actor = actor
        self.adaptation = adaptation

    def forward(self, observation, history):
        latent = self.adaptation(history)
        return torch.tanh(self.actor(torch.cat([observation, latent], dim=-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-dir', default='outputs/third_round/seed_0')
    parser.add_argument('--out', default='outputs/third_round/deployment')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(24680)
    checks = {}
    for name, folder in [('baseline', 'baseline'), ('cnn', 'pooled_cnn'), ('ordered', 'ordered_mlp')]:
        source = Path(args.seed_dir) / folder / 'latest.pt'
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        policy = Policy('baseline' if name == 'baseline' else 'teacher')
        policy.load_state_dict(checkpoint['policy'])
        policy.eval().requires_grad_(False)
        obs = torch.randn(16, OBS_DIM)
        history = torch.randn(16, 50, OBS_DIM)
        history[0] = 0
        history[0, -1] = obs[0]
        adapter = None
        if name == 'baseline':
            module = BaselineDeployment(policy.actor)
            inputs = (obs,)
        else:
            adapter = make_adaptation(checkpoint['config'].get('architecture', 'pooled_cnn'), 50)
            adapter.load_state_dict(checkpoint['adaptation'])
            adapter.eval().requires_grad_(False)
            module = RMADeployment(policy.actor, adapter)
            inputs = (obs, history)
        module.eval()
        scripted = torch.jit.freeze(torch.jit.script(module))
        filename = out / (name + '.torchscript.pt')
        scripted.save(str(filename))
        loaded = torch.jit.load(str(filename))
        with torch.inference_mode():
            expected = policy.deterministic(obs) if adapter is None else policy.deterministic(obs, latent=adapter(history))
            actual = loaded(*inputs)
            torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
            assert actual.shape == (16,12) and actual.abs().max() <= 1
            # Batch-one cold start must also remain supported.
            one_inputs = tuple(value[:1] for value in inputs)
            torch.testing.assert_close(module(*one_inputs), loaded(*one_inputs), atol=1e-6, rtol=1e-6)
        checks[name] = dict(file=filename.name, source=str(source.resolve()),
                           source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                           max_abs_difference=float((expected-actual).abs().max()),
                           input_shapes=[list(value.shape) for value in inputs], parameters=sum(p.numel() for p in module.parameters()))
    manifest = dict(torch=torch.__version__, models=checks, device='cpu', mode='deterministic',
        observation_layout=['local angular velocity:3','projected gravity:3','velocity command:3',
                            'joint position minus nominal:12','joint velocity * 0.05:12','previous normalized action:12'],
        history='[batch,50,45], oldest to newest; clear on episode reset, new observation in final frame',
        output='[batch,12] in [-1,1]; q_target=clip(q_nominal+0.5*action, joint limits)',
        control='50Hz position targets; simulation affine PD kp=30,kd=0.8 at 500Hz, scaled motor strength and torque limit',
        note='Exports contain only actor and optional adaptation. No privileged encoder or critic. Real robot sensor preprocessing, joint mapping, communication and safety are not implemented or validated.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
