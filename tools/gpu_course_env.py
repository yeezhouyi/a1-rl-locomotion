"""A1 obstacle course extension: explicit task/reset/reward changes."""
import jax
import jax.numpy as jp
from mujoco import mjx
from gpu_a1_env import A1Joystick


def ground_height(x,y):
    up=jp.clip(.12*x,0,.24)
    down=jp.clip(.12*(5.2-x),0,.24)
    ramp=jp.minimum(up,down)
    stairs_up=jp.clip(jp.floor(x/.4)+1,0,6)*.035
    stairs_down=jp.clip(6-jp.floor((x-3.2)/.4),0,6)*.035
    stairs=jp.minimum(stairs_up,stairs_down)
    return jp.where(jp.abs(y-3)<=1.2,ramp,jp.where(jp.abs(y+3)<=1.2,stairs,0.))


class A1Course(A1Joystick):
    def __init__(self,asset_dir,batch_size=512,noise=1.0,reward_profile='locomotion_v2',**kwargs):
        super().__init__(asset_dir,batch_size=batch_size,noise=noise,terrain='course',reward_profile=reward_profile)
        # Elevated support surfaces invalidate absolute foot-z swing rewards.
        self._config.reward_config.scales.feet_height=0.
        self._config.njmax=640
        self._config.naconmax=96*batch_size
        with self._config.unlocked():
            self._config.course_reset_x=[-1.5,5.8]
            self._config.course_lane_y=[-3.,0.,3.]
            self._config.course_cmd_vx=[.25,.65]
            self._config.course_cmd_stop_probability=.1
            self._config.course_cmd_yaw_max=.08

    def reset(self,rng):
        state=super().reset(rng)
        keys=jax.random.split(jax.random.fold_in(rng,610),5)
        lane=jax.random.randint(keys[0],(),0,3)
        lane_y=jp.array([-3.,0.,3.])[lane]
        x=jax.random.uniform(keys[1],(),minval=-1.5,maxval=5.8)
        y=lane_y+jax.random.uniform(keys[2],(),minval=-.25,maxval=.25)
        yaw=jax.random.uniform(keys[3],(),minval=-.2,maxval=.2)
        quat=jp.array([jp.cos(yaw/2),0.,0.,jp.sin(yaw/2)])
        q=state.data.qpos.at[:3].set(jp.array([x,y,.35+ground_height(x,y)]))
        q=q.at[3:7].set(quat)
        data=mjx.forward(self.mjx_model,state.data.replace(qpos=q))
        state.info['command']=self.sample_command(keys[4],jp.zeros(3))
        state.info['lane_y']=lane_y
        return state.replace(data=data,obs=self._get_obs(data,state.info))

    def sample_command(self,rng,x_k):
        keys=jax.random.split(rng,4)
        vx=jax.random.uniform(keys[0],(),minval=.25,maxval=.65)
        vx=jp.where(jax.random.bernoulli(keys[1],.1),0.,vx)
        vy=jax.random.uniform(keys[2],(),minval=-.03,maxval=.03)
        yaw=jax.random.uniform(keys[3],(),minval=-.08,maxval=.08)
        return jp.array([vx,vy,yaw])

    def _cost_feet_clearance(self,data):
        feet=data.site_xpos[self._feet_site_id]
        relative_z=feet[:,2]-ground_height(feet[:,0],feet[:,1])
        vel=data.sensordata[self._foot_linvel_sensor_adr][:,:2]
        return jp.sum(jp.abs(relative_z-self._config.reward_config.max_foot_height)*jp.sqrt(jp.linalg.norm(vel,axis=-1)))

    def step(self,state,action):
        state=super().step(state,action)
        q=state.data.qpos
        outside=(jp.abs(q[1]-state.info['lane_y'])>1.1)|(q[0]<-2.)|(q[0]>9.)
        clearance=q[2]-ground_height(q[0],q[1])
        return state.replace(done=jp.maximum(state.done,(outside|(clearance<.14)).astype(state.done.dtype)))
