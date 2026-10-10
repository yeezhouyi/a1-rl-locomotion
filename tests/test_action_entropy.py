import torch
from torch.distributions import Normal
from rma_local.ppo import squashed_entropy, tanh_log_jacobian


def test_stable_jacobian_matches_identity_and_handles_saturation():
    raw = torch.tensor([-2., -0.5, 0., 0.5, 2.], dtype=torch.float64)
    torch.testing.assert_close(tanh_log_jacobian(raw), torch.log1p(-raw.tanh().square()))
    extreme = torch.tensor([-100., 100.], requires_grad=True)
    jacobian = tanh_log_jacobian(extreme)
    assert torch.isfinite(jacobian).all()
    jacobian.sum().backward()
    torch.testing.assert_close(extreme.grad, torch.tensor([2., -2.]))


def test_action_entropy_detects_saturation_at_equal_gaussian_entropy():
    noise = torch.linspace(-3., 3., 4096).reshape(-1, 1)
    centered = Normal(torch.zeros_like(noise), torch.full_like(noise, 0.5))
    mean = torch.full_like(noise, 8., requires_grad=True)
    saturated = Normal(mean, torch.full_like(noise, 0.5))
    torch.testing.assert_close(centered.entropy(), saturated.entropy())
    assert squashed_entropy(saturated, noise).mean() < squashed_entropy(centered, noise).mean() - 10
    (-squashed_entropy(saturated, noise).mean()).backward()
    assert torch.isfinite(mean.grad).all() and mean.grad.sum() > 1.9
