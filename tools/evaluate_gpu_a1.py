"""Fixed-command evaluation of a saved upstream PPO A1 checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jp
from gpu_checkpoint import load_policy
from mujoco import mjx
import numpy as np
from gpu_a1_env import A1Joystick


def evaluate(env, inference_fn, out, batch=8, horizon=1000, protocol="development"):
    if jax.default_backend() != "gpu":
        raise RuntimeError("Independent evaluation requires the actual GPU backend")
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    infer = jax.jit(jax.vmap(inference_fn))
    # Four deterministic scenarios, two reset seeds each. Pair across models.
    scenarios=[("slow",[.3,0,0]),("fast",[.65,0,0]),("stop_resume",[.4,0,0]),("turn_reversal",[.4,0,0])]
    scenarios= scenarios if protocol=="development" else [("slow",[.22,0,0]),("fast",[.5,0,0]),("stop_resume",[.35,0,0]),("turn_reversal",[.3,0,0])]
    seed_start={"development":82000,"holdout":93000,"confirmation":104000,"spatial_holdout":116000}[protocol]
    seeds=np.arange(seed_start,seed_start+8)
    commands=np.repeat(np.array([x[1] for x in scenarios],np.float32),2,axis=0)
    state=reset(jax.vmap(jax.random.PRNGKey)(jp.array(seeds)))
    qpos=jp.tile(env._init_q,(8,1))
    # 0.5m/s for 20s needs 10m, exceeding the 9.5m safety boundary
    # from zero. Freeze a confirmation layout for both checkpoints.
    origin_x=-2.0 if protocol in ("confirmation","spatial_holdout") and "rough" in env.xml_path else 0.0
    qpos=qpos.at[:,0].set(origin_x+jp.linspace(-.04,.04,8))
    # Independent, reproducible joint offsets avoid counting translated copies
    # of the same deterministic flat-ground rollout as independent robustness.
    joint_offsets=np.stack([np.random.default_rng(int(seed)).uniform(-.03,.03,12) for seed in seeds])
    qpos=qpos.at[:,7:].add(jp.array(joint_offsets,dtype=jp.float32))
    data=state.data.replace(qpos=qpos,qvel=jp.zeros((8,env.mj_model.nv)),ctrl=qpos[:,7:])
    data=jax.jit(jax.vmap(lambda d:mjx.forward(env.mjx_model,d)))(data)
    state=state.replace(data=data)
    state.info["last_act"]=jp.zeros((8,12));state.info["command"]=jp.array(commands)
    state.info["steps_until_next_cmd"]=jp.full((8,),1_000_000,dtype=jp.int32)
    obs=jax.jit(jax.vmap(env._get_obs))(state.data,state.info)
    state=state.replace(obs=obs)
    valid=np.ones(8,bool); lengths=np.zeros(8,int);falls=np.zeros(8,bool)
    outside=np.zeros(8,bool)
    histories=[];rng=jax.random.PRNGKey(13)
    get_velocity=jax.jit(jax.vmap(lambda d:jp.hstack((env.get_local_linvel(d)[:2],env.get_gyro(d)[2:3]))))
    start=time.perf_counter()
    for t in range(horizon):
        if t==200:commands[4:6]=[.65,0,0];commands[6:8]=[.4,0,.3]
        if t==450:commands[4:6]=[0,0,0];commands[6:8]=[.4,0,-.3]
        if t==600:commands[4:6]=[.4,0,0]
        if t==700:commands[6:8]=[.4,0,0]
        if protocol in ("holdout","confirmation","spatial_holdout"):
            if t==200:commands[4:6]=[.5,0,0];commands[6:8]=[.3,0,-.2]
            if t==450:commands[4:6]=[0,0,0];commands[6:8]=[.3,0,.2]
            if t==600:commands[4:6]=[.35,0,0]
            if t==700:commands[6:8]=[.3,0,0]
        state.info["command"]=jp.array(commands)
        state.info["steps_until_next_cmd"]=jp.full((8,),1_000_000,dtype=jp.int32)
        state=state.replace(obs={k:v.at[:,-3:].set(jp.array(commands)) if k=="state" else v for k,v in state.obs.items()})
        rng,key=jax.random.split(rng);actions=infer(state.obs,jax.random.split(key,8))[0]
        state=step(state,actions)
        velocity=np.asarray(get_velocity(state.data));q=np.asarray(state.data.qpos);v=np.asarray(state.data.qvel)
        is_fall=np.asarray(state.done).astype(bool) | (q[:,2]<.14) | ~np.isfinite(q).all(axis=1)
        out_of_bounds=(np.abs(q[:,:2])>9.5).any(axis=1) if "rough" in env.xml_path else np.zeros(8,bool)
        outside |= valid & out_of_bounds
        lengths[valid]+=1;falls |= valid & is_fall
        histories.append(dict(qpos=q.copy(),qvel=v.copy(),velocity=velocity.copy(),command=commands.copy(),valid=valid.copy(),action=np.asarray(actions)))
        valid &= ~(is_fall | out_of_bounds)
    arrays={k:np.stack([x[k] for x in histories]) for k in histories[0]}
    np.savez_compressed(out/"trajectories.npz",**arrays)
    rows=[]
    for i in range(8):
        count=lengths[i];error=arrays['velocity'][:count,i]-arrays['command'][:count,i]
        vel=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1))));yaw=float(np.sqrt(np.mean(error[:,2]**2)))
        rows.append(dict(scenario=scenarios[i//2][0],seed=int(seeds[i]),steps=int(count),fell=bool(falls[i]),out_of_bounds=bool(outside[i]),velocity_rmse=vel,yaw_rmse=yaw,
            tracking_success=bool(count==horizon and not falls[i] and not outside[i] and vel<.2 and yaw<.3),final_x=float(arrays['qpos'][count-1,i,0]),
            initial_x=float(qpos[i,0]),final_y=float(arrays['qpos'][count-1,i,1])))
    summary=dict(episodes=rows,tracking_successes=sum(x['tracking_success'] for x in rows),falls=sum(x['fell'] for x in rows),
        elapsed_s=time.perf_counter()-start,backend=jax.default_backend(),devices=[str(x) for x in jax.devices()],
        horizon_steps=horizon,control_dt=env.dt,terrain=Path(env.xml_path).stem,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_joint_offset_rad=.03,
        protocol=protocol,out_of_bounds=sum(x['out_of_bounds'] for x in rows),
        origin_x=origin_x,env_config=env._config.to_dict(),
        environment_sha256=hashlib.sha256(Path(__file__).with_name("gpu_a1_env.py").read_bytes()).hexdigest(),
        note="8 fixed evaluation episodes with seed-specific initial joint offsets, one trained seed; GPU physics. Independent command schedule, no automatic resets. Fall adds height<0.14m to upstream flip termination.")
    (out/"evaluation.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--assets",required=True);p.add_argument("--out",required=True);p.add_argument("--terrain",default="flat")
    p.add_argument("--reward-profile",choices=["upstream","locomotion_v2"],default="upstream")
    p.add_argument("--protocol",choices=["development","holdout","confirmation","spatial_holdout"],default="development")
    args=p.parse_args()
    env=A1Joystick(args.assets,batch_size=8,terrain=args.terrain,noise=0.0,reward_profile=args.reward_profile)
    inference,network_config=load_policy(args.checkpoint,deterministic=True)
    evaluate(env,inference,args.out,protocol=args.protocol)
    (Path(args.out)/"checkpoint_source.json").write_text(json.dumps(dict(checkpoint=args.checkpoint,
        network_config=network_config),indent=2))
