"""Generate focused local portfolio page using existing measured video files."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('<EVIDENCE>'))
    root=p.parse_args().root
    directory=root/'showcase';directory.mkdir(exist_ok=True)
    trials=[('course_trained_heading_holdout','地形专门续训','course_trained_demo'),('course_transfer_heading_holdout','已有粗糙策略直接迁移','course_transfer_demo')]
    for name,_,demo in trials:
        assert (root/name/'result/evaluation.json').is_file()
        assert (directory/demo/'demo_manifest.json').is_file()
    data=json.loads((root/trials[0][0]/'result/evaluation.json').read_text())
    labels={'stairs_straight':'上下六级台阶','stairs_stop_resume':'台阶变速与停走','slope_straight':'完整上坡与下坡','slope_turn':'坡地转向'}
    clips=[]
    manifest=json.loads((root/trials[0][0]/'showcase/video_manifest.json').read_text())
    for shot in manifest:
        row=data['episodes'][shot['episode_index']]
        clip=f'../{trials[0][0]}/showcase/{shot["file"]}'
        clips.append(f'<article><h3>{labels[shot["scenario"]]}</h3><video controls preload="metadata" src="{clip}"></video><p>seed {row["seed"]} · 20秒完整穿越 · 速度RMSE {row["velocity_rmse"]:.3f} m/s</p></article>')
    rows=[]
    for name,label,_ in trials:
        summary=json.loads((root/name/'result/evaluation.json').read_text())
        rows.append(f'<tr><td>{label}</td><td>{summary["tracking_successes"]}/8</td><td>{summary["falls"]}/8</td><td>{summary["out_of_bounds"]}/8</td><td><a href="../{name}/result/evaluation.json">逐回合指标</a></td></tr>')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A1 强化学习 · 坡地与台阶视频</title><style>*{box-sizing:border-box}body{margin:0;background:#0c1420;color:#e1edf7;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1140px;margin:auto;padding:32px 20px 60px}h1{font-size:clamp(30px,5vw,46px);line-height:1.2;margin:16px 0}h2{font-size:26px}p{color:#abc2d5}a{color:#60ddbe}.eyebrow{color:#60ddbe;font-size:13px;letter-spacing:.1em}.chips{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0}.chips span{border:1px solid #314558;padding:5px 13px;border-radius:24px}.panel{padding:24px;background:#121f2e;border:1px solid #2b4054;border-radius:18px;margin:24px 0}video{width:100%;background:#070d14;border-radius:12px;aspect-ratio:16/9}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}h3{margin-top:0}.actions{display:flex;flex-wrap:wrap;gap:18px;margin-top:16px}.button{padding:9px 18px;background:#193d3a;border:1px solid #3c9380;border-radius:9px;text-decoration:none}.note{font-size:14px}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #314558}.scroll{overflow:auto}@media(max-width:720px){.grid{grid-template-columns:1fr}.panel{padding:16px}main{padding:24px 14px}table{min-width:650px}}</style></head><body><main><div class="eyebrow">UNITREE A1 · LOCAL GPU REINFORCEMENT LEARNING</div><h1>从速度跟踪<br>到完整穿越坡地与台阶</h1><p>实际在 RTX 4060 Laptop 上训练与评估。沿用 MuJoCo Playground 和 Brax PPO，新增真实碰撞地形，结合参考 legged_gym 的高层航向指令反馈。</p><div class="chips"><span>六级 3.5cm 台阶</span><span>6.84° 上下坡</span><span>变速 · 停走 · 转向</span><span>固定保留集 8/8</span></div><section class="panel"><h2>90秒完整任务演示</h2><video controls preload="metadata" poster="course_trained_demo/poster.jpg" src="course_trained_demo/A1_terrain_demo.mp4"></video><div class="actions"><a class="button" href="course_trained_demo/A1_terrain_demo.mp4" download>下载 MP4 · 1280×720</a><a href="course_trained_demo/demo_manifest.json">片段清单与SHA-256</a><a href="../index.html">完整实验与原始证据</a></div><p class="note">每类任务取首个达标回合，四段均保留完整20秒；1×真实时间。画面为GPU评估轨迹回放，无自动重置。每帧显示指令和实际速度；无音轨。</p></section><section class="panel"><h2>20秒单场景回放</h2><div class="grid">'''+''.join(clips)+'''</div></section><section class="panel"><h2>同条件对照</h2><div class="scroll"><table><tr><th>冻结模型</th><th>达标</th><th>跌倒</th><th>越界</th><th>证据</th></tr>'''+''.join(rows)+'''</table></div><p>两模型均使用相同高层航向反馈、初始seed、几何、指令与评估脚本，全部跨过上/下障碍。旧模型也8/8，因此不声称专门续训提高通过率。</p><details><summary>查看旧粗糙模型直接迁移的视频</summary><video controls preload="metadata" poster="course_transfer_demo/poster.jpg" src="course_transfer_demo/A1_terrain_demo.mp4"></video><a href="course_transfer_demo/A1_terrain_demo.mp4" download>下载对照视频</a></details></section><section class="panel"><h2>控制与结果边界</h2><p>底层为48维观测PPO速度策略；高层利用仿真机体yaw与横向位置生成yaw-rate指令，并不修改关节动作或物理状态。专门续训纯yaw-rate开发集为4/8，仍有通道偏离；反馈是完整演示的组成部分。</p><p>达标要求完整20秒、未跌倒/越界、最终x≥6m，速度向量RMSE&lt;0.2m/s及yawRMSE&lt;0.3rad/s。单训练seed、8个固定回合、小规模仿真验证，不外推为更高台阶、大坡度、真机或跨seed稳定性。</p><p class="note">代码、配置、模型恢复验证和失败日志均保留；大文件位于D盘。<a href="../course_delivery_verification.json">交付核验记录</a></p></section></main></body></html>'''
    (directory/'index.html').write_text(page,encoding='utf-8')
    print(directory/'index.html')


if __name__=='__main__':main()
