"""Compose full measured episodes; selected shots and aggregate remain explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def build(result,media,out):
    out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((result/'evaluation.json').read_text())
    manifest=json.loads((media/'video_manifest.json').read_text())
    selected=[]
    for scenario in ('stairs_straight','stairs_stop_resume','slope_straight','slope_turn'):
        options=[x for x in manifest if x['scenario']==scenario]
        passed=[x for x in options if x['passed']]
        if not passed:raise RuntimeError(f'No complete passed clip for {scenario}; do not advertise unfinished traversal')
        selected.append(passed[0])
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',46)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    def card(end=False):
        im=Image.new('RGB',(1280,720),(12,20,32));d=ImageDraw.Draw(im)
        d.text((80,95),'A1 四足强化学习 · 坡地与台阶' if not end else '完整穿越结果与复现边界',font=title,fill=(225,237,246))
        lines=['六级3.5cm台阶 / 6.84°上下坡 / 变速停走与转向',
               'MuJoCo Playground + Brax PPO · RTX 4060 Laptop',
               '底层速度策略 + 高层航向反馈' if summary.get('heading_feedback') else 'PPO速度策略',
               f"本组固定保留集 {summary['tracking_successes']}/8 达标 · {summary['falls']}/8 跌倒",
               '每段展示完整20秒成功回合，1×真实时间' if not end else '展示片段按每场景首个达标回合选取，完整指标单独保留',
               '单训练种子 · 仿真 · 无真机验证']
        for j,line in enumerate(lines):d.text((80,220+62*j),line,font=font if j<4 else small,fill=(96,221,190) if j==3 else (174,197,215))
        return np.asarray(im)
    video=out/'A1_terrain_demo.mp4'
    with iio.get_writer(video,fps=25,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-movflags','+faststart']) as writer:
        for _ in range(125):writer.append_data(card())
        for shot in selected:
            with iio.get_reader(media/shot['file']) as reader:
                for frame in reader:writer.append_data(frame)
        for _ in range(125):writer.append_data(card(True))
    Image.fromarray(card()).save(out/'poster.jpg',quality=95)
    record=dict(file=video.name,aggregate_success=summary['tracking_successes'],aggregate_falls=summary['falls'],protocol=summary['protocol'],
                heading_feedback=summary.get('heading_feedback',False),shots=selected,speed=1,
                note='Selected first passed episode per scenario. Not the complete 8-episode evaluation; see linked metrics.',
                sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (out/'demo_manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--media',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.result,a.media,a.out)
