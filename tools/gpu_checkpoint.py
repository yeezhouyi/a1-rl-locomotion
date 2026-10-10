"""Restore Brax 0.14.2 PPO without its nullable-initializer loader bug.

Uses saved architecture, observation normalization and Orbax parameters.
Upstream save writes mean_kernel_init_fn=null; upstream load_config indexes
KERNEL_INITIALIZER[None]. Preserve None, its documented network default.
"""
import json
from pathlib import Path
from brax.training import networks as base_networks
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks


def load_policy(path, deterministic=True):
    config=json.loads((Path(path)/'ppo_network_config.json').read_text())
    kwargs=dict(config['network_factory_kwargs'])
    kwargs['activation']=base_networks.ACTIVATION[kwargs['activation']]
    for key in ('policy_network_kernel_init_fn','value_network_kernel_init_fn','mean_kernel_init_fn'):
        if kwargs.get(key) is not None:
            kwargs[key]=base_networks.KERNEL_INITIALIZER[kwargs[key]]
    sizes={}
    for key,spec in config['observation_size'].items():
        if isinstance(spec,dict):
            shape=spec['shape']
        elif isinstance(spec,int):
            shape=[spec]
        else:
            shape=spec
        if len(shape)!=1:raise ValueError('Only saved vector observations are supported')
        sizes[key]=int(shape[0])
    preprocess=running_statistics.normalize if config['normalize_observations'] else (lambda x,y:x)
    network=networks.make_ppo_networks(sizes,config['action_size'],preprocess_observations_fn=preprocess,**kwargs)
    params=checkpoint.load(path)
    policy=networks.make_inference_fn(network)(params,deterministic=deterministic)
    return policy,config
