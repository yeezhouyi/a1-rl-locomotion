"""Render recorded GPU traversals with readable Chinese telemetry, no resimulation."""
import argparse
import json
from pathlib import Path
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LABELS={'stairs_straight':'六级台阶 · 上下台阶', 'stairs_stop_resume':'六级台阶 · 变速与停走',
        'slope_straight':'6.8°坡地 · 上坡与下坡', 'slope_turn':'6.8°坡地 · 转向指令'}
BG=(12,20,32);FG=(225,235,245);MUTED=(160,181,200);ACCENT=(92,222,187)
FONT='C:/Windows/Fonts/msyh.ttc';BOLD='C:/Windows/Fonts/msyhbd.ttc'


def render(assets,result,out,episode_indices=None):
    out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((result/'evaluation.json').read_text())
    arrays=np.load(result/'trajectories.npz')
    model=mujoco.MjModel.from_xml_path(str(assets/'a1_course.xml'))
    model.vis.global_.offwidth=960;model.vis.global_.offheight=600
    model.vis.headlight.ambient[:]=.4;model.vis.headlight.diffuse[:]=.7
    data=mujoco.MjData(model)
    font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,18);title=ImageFont.truetype(BOLD,28)
    def text(draw,xy,content,color=FG,f=font):draw.text(xy,content,font=f,fill=color)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance=2.7;cam.azimuth=135;cam.elevation=-23
    manifest=[]
    with mujoco.Renderer(model,height=600,width=960) as renderer:
        for i,ep in enumerate(summary['episodes']):
            if episode_indices is not None and i not in episode_indices:continue
            filename=f"{ep['scenario']}_seed{ep['seed']}.mp4"
            n=ep['steps'];lane=-3 if i<4 else 3
            with iio.get_writer(out/filename,fps=25,codec='libx264',quality=8,
                                macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
                for frame in range(0,n,2):
                    data.qpos[:]=arrays['qpos'][frame,i];data.qvel[:]=arrays['qvel'][frame,i]
                    mujoco.mj_forward(model,data)
                    cam.lookat[:]=data.qpos[:3]+np.array([.4,0.,-.05])
                    renderer.update_scene(data,camera=cam)
                    canvas=Image.new('RGB',(1280,720),BG);canvas.paste(Image.fromarray(renderer.render()),(0,80))
                    draw=ImageDraw.Draw(canvas)
                    text(draw,(24,15),LABELS[ep['scenario']],f=title)
                    controller='PPO + 航向反馈' if summary.get('heading_feedback') else 'PPO速度策略'
                    text(draw,(24,51),f"Unitree A1  |  {controller}  |  seed {ep['seed']}  |  1×真实时间",MUTED,small)
                    command=arrays['command'][frame,i];velocity=arrays['velocity'][frame,i]
                    t=(frame+1)*.02
                    text(draw,(985,94),f'回合时间  {t:5.2f} s')
                    text(draw,(985,138),'前进速度 / m/s',MUTED,small)
                    text(draw,(985,173),f'指令 {command[0]:.2f}',ACCENT)
                    text(draw,(985,208),f'实际 {velocity[0]:.2f}')
                    text(draw,(985,261),'偏航速度 / rad/s',MUTED,small)
                    text(draw,(985,296),f'指令 {command[2]:+.2f}',ACCENT)
                    text(draw,(985,331),f'实际 {velocity[2]:+.2f}')
                    text(draw,(985,389),'完整穿越要求',MUTED,small)
                    text(draw,(985,422),'越过障碍并抵达 x ≥ 6m',f=small)
                    progress=float(data.qpos[0]);fraction=np.clip((progress+1.3)/7.3,0,1)
                    draw.rounded_rectangle((985,465,1250,478),radius=5,fill=(48,65,80))
                    draw.rounded_rectangle((985,465,985+max(2,265*fraction),478),radius=5,fill=ACCENT)
                    text(draw,(985,495),f'当前 x = {progress:.2f} m',f=small)
                    text(draw,(985,538),'台阶 3.5cm × 6级' if i<4 else '坡度 6.84° / 峰高24cm',MUTED,small)
                    if frame+2>=n:
                        status='达标：完整20秒并穿越' if ep['tracking_success'] else '未达标：跌倒' if ep['fell'] else '未达标：越界' if ep['out_of_bounds'] else '未达标：跟踪或穿越不足'
                        text(draw,(985,590),status,ACCENT if ep['tracking_success'] else (255,128,112),small)
                    text(draw,(24,690),'GPU策略记录回放 · 无自动重置 · 单训练种子 · 仿真结果',MUTED,small)
                    writer.append_data(np.asarray(canvas))
                    if frame in (0,250,500,750):canvas.save(out/f'{ep["scenario"]}_seed{ep["seed"]}_frame{frame}.jpg',quality=90)
            manifest.append(dict(file=filename,episode_index=i,seed=ep['seed'],scenario=ep['scenario'],
                                 passed=ep['tracking_success'],fell=ep['fell'],duration_s=((n+1)//2)/25))
    (out/'video_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',choices=range(8))
    a=p.parse_args();render(a.assets,a.result,a.out,a.episodes)
