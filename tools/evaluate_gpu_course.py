"""Fixed complete traversals; success requires crossing actual obstacles."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
from gpu_course_env import A1Course,ground_height
from gpu_checkpoint import load_policy


def evaluate(env,inference,out,protocol,heading_feedback=False):
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    seed_start=127000 if protocol=='development' else 138000
    seeds=np.arange(seed_start,seed_start+8)
    lanes=np.array([-3]*4+[3]*4,dtype=np.float32)
    names=['stairs_straight','stairs_stop_resume','slope_straight','slope_turn']
    reset=jax.jit(jax.vmap(env.reset));step=jax.jit(jax.vmap(env.step));infer=jax.jit(jax.vmap(inference))
    get_obs=jax.jit(jax.vmap(env._get_obs));termination=jax.jit(jax.vmap(env._get_termination))
    state=reset(jax.vmap(jax.random.PRNGKey)(jp.array(seeds)))
    q=jp.tile(env._init_q,(8,1)).at[:,0].set(-1.3)
    q=q.at[:,1].set(jp.array(lanes)+jp.linspace(-.06,.06,8)).at[:,2].set(.35)
    offsets=np.stack([np.random.default_rng(int(s)).uniform(-.03,.03,12) for s in seeds])
    q=q.at[:,7:].add(jp.array(offsets,dtype=jp.float32))
    data=state.data.replace(qpos=q,qvel=jp.zeros((8,env.mj_model.nv)),ctrl=q[:,7:])
    data=jax.jit(jax.vmap(lambda d:mjx.forward(env.mjx_model,d)))(data)
    state=state.replace(data=data)
    state.info['lane_y']=jp.array(lanes)
    velocity=jax.jit(jax.vmap(lambda d:jp.hstack((env.get_local_linvel(d)[:2],env.get_gyro(d)[2:3]))))
    valid=np.ones(8,bool);lengths=np.zeros(8,int);falls=np.zeros(8,bool);outside=np.zeros(8,bool)
    hist=[];started=time.perf_counter();key=jax.random.PRNGKey(713)
    desired_heading=np.zeros(8,np.float32)
    for t in range(1000):
        commands=np.tile(np.array([.5,0.,0.],np.float32),(8,1))
        commands[2:4,0]=.45 if t<200 else .6 if t<350 else 0. if t<450 else .5 if t<700 else .45
        commands[6:8,2]=.12 if 300<=t<400 else -.12 if 500<=t<600 else 0.
        if heading_feedback:
            # Borrow legged_gym's target-heading -> yaw-rate command interface.
            # Explicit high-level feedback, not an extra learned skill or teleport.
            desired_heading += commands[:,2]*.02
            current_q=np.asarray(state.data.qpos)
            qw,qx,qy,qz=current_q[:,3:7].T
            heading=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
            target=desired_heading+.5*(lanes-current_q[:,1])
            error=(target-heading+np.pi)%(2*np.pi)-np.pi
            commands[:,2]=np.clip(commands[:,2]+1.5*error,-.4,.4)
        state.info['command']=jp.array(commands)
        state.info['steps_until_next_cmd']=jp.full((8,),1000000,dtype=jp.int32)
        state=state.replace(obs=get_obs(state.data,state.info))
        key,k=jax.random.split(key);actions=infer(state.obs,jax.random.split(k,8))[0]
        state=step(state,actions)
        qpos=np.asarray(state.data.qpos);vel=np.asarray(velocity(state.data))
        support=np.asarray(ground_height(jp.array(qpos[:,0]),jp.array(qpos[:,1])))
        is_fall=(qpos[:,2]-support<.14)|~np.isfinite(qpos).all(axis=1)|np.asarray(termination(state.data)).astype(bool)
        # The course ends at x=5.8m; the exit is an infinite physical plane.
        # Keep enough exit distance for the full 20s command schedule.
        is_out=(np.abs(qpos[:,1]-lanes)>1.1)|(qpos[:,0]<-2)|(qpos[:,0]>12)
        falls|=valid&is_fall;outside|=valid&is_out;lengths[valid]+=1
        hist.append(dict(qpos=qpos.copy(),qvel=np.asarray(state.data.qvel),velocity=vel.copy(),command=commands.copy(),valid=valid.copy(),action=np.asarray(actions)))
        valid &= ~(is_fall|is_out)
    arrays={k:np.stack([h[k] for h in hist]) for k in hist[0]}
    np.savez_compressed(out/'trajectories.npz',**arrays)
    episodes=[]
    for i in range(8):
        n=int(lengths[i]);error=arrays['velocity'][:n,i]-arrays['command'][:n,i]
        rmse=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1))));yaw=float(np.sqrt(np.mean(error[:,2]**2)))
        final_x=float(arrays['qpos'][n-1,i,0]);crossed=final_x>=6.0
        episodes.append(dict(scenario=names[i//2],seed=int(seeds[i]),steps=n,fell=bool(falls[i]),out_of_bounds=bool(outside[i]),
            velocity_rmse=rmse,yaw_rmse=yaw,final_x=final_x,initial_x=-1.3,final_y=float(arrays['qpos'][n-1,i,1]),
            crossed_obstacles=crossed,tracking_success=bool(n==1000 and not falls[i] and not outside[i] and crossed and rmse<.2 and yaw<.3)))
    summary=dict(episodes=episodes,tracking_successes=sum(x['tracking_success'] for x in episodes),falls=int(falls.sum()),out_of_bounds=int(outside.sum()),
        protocol=protocol,heading_feedback=heading_feedback,terrain='a1_course',horizon_steps=1000,control_dt=.02,elapsed_s=time.perf_counter()-started,
        initial_joint_offset_rad=.03,backend=jax.default_backend(),env_config=env._config.to_dict(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment_sha256=hashlib.sha256(Path(__file__).with_name('gpu_course_env.py').read_bytes()).hexdigest(),
        note='Fixed physical stairs/ramp lanes. Success requires final x>=6m, whole 20s and remaining within lane. Exit safety x<=12m; obstacle ends x=5.8m. One training seed.')
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--terrain',default='course');p.add_argument('--reward-profile',default='locomotion_v2');p.add_argument('--protocol',default='development')
    p.add_argument('--heading-feedback',action='store_true')
    a=p.parse_args();env=A1Course(a.assets,batch_size=8,noise=0.,reward_profile=a.reward_profile)
    inference,config=load_policy(a.checkpoint,deterministic=True)
    evaluate(env,inference,a.out,a.protocol,a.heading_feedback)
    (Path(a.out)/'checkpoint_source.json').write_text(json.dumps(dict(checkpoint=a.checkpoint,network_config=config),indent=2))
