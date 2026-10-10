"""Build a compact evidence page from completed GPU runs on D:."""
import argparse
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('<EVIDENCE>'))
    args=p.parse_args();root=args.root
    probes=[];trains=[];evaluations=[];failures=[]
    preferred=['course_trained_heading_holdout','course_transfer_heading_holdout','course_trained_raw_dev','course_transfer_dev_v3','rough_wider_spatial_holdout','rough_transfer_spatial_holdout','rough_wider_confirmation','rough_wider_flat_holdout','rough_finetuned_confirmation','rough_transfer_confirmation','flat_rewardv2_holdout','rough_finetuned_flat_holdout','rough_finetuned_holdout_capacity320','rough_transfer_holdout_capacity320']
    for directory in sorted(root.iterdir(),key=lambda d:(preferred.index(d.name) if d.name in preferred else len(preferred),d.name)):
        if not directory.is_dir():continue
        result=directory/'result';execution=directory/'execution_summary.json'
        if not execution.is_file():continue
        status=json.loads(execution.read_text());name=html.escape(directory.name)
        if status['exit_code']!=0:
            failures.append(f'<li>{name}：退出码 {status["exit_code"]}，不计入有效训练结果。<a href="{name}/console.log">日志</a></li>')
            continue
        probe=result/'probe.json'
        if probe.is_file():
            data=json.loads(probe.read_text())
            probes.append(f'<tr><td>{name}</td><td>{data["versions"]["jax"]}</td><td>{data["batch"]}</td><td>{data["control_steps_per_s"]:,.0f}</td><td>{data["reset_compile_s"]:.2f} / {data["step_compile_s"]:.2f}</td><td>{status["peak_total_device_used_mib"]}</td></tr>')
        completion=result/'completion.json'
        if completion.is_file():
            data=json.loads(completion.read_text());steps=max(data['observed_callbacks'])
            trains.append(f'<p><b>{name}</b>：{steps:,} 控制采样步，{data["elapsed_s"]:.1f} 秒；整卡峰值 {status["peak_total_device_used_mib"]} MiB。<a href="{name}/result/config.json">配置</a> · <a href="{name}/result/progress.jsonl">进度原始数据</a></p>')
        evaluation=result/'evaluation.json'
        if evaluation.is_file():
            data=json.loads(evaluation.read_text())
            rows=''.join(f'<tr><td>{html.escape(x["scenario"])}</td><td>{x["seed"]}</td><td>{x["steps"]}</td><td>{x["velocity_rmse"]:.3f}</td><td>{x["yaw_rmse"]:.3f}</td><td>{"跌倒" if x["fell"] else "越界" if x.get("out_of_bounds") else "存活"}</td><td>{"通过" if x["tracking_success"] else "未通过"}</td></tr>' for x in data['episodes'])
            media=directory/'media'
            videos=''.join(f'<figure><figcaption>{html.escape(f.stem)}</figcaption><video controls preload="metadata" src="{name}/media/{f.name}"></video></figure>' for f in sorted(media.glob('*.mp4')))
            plot=f'<img src="{name}/media/tracking.png" alt="指令与实际速度">' if (media/'tracking.png').exists() else ''
            perturbation='含固定seed关节初始偏移' if data.get('initial_joint_offset_rad') else '早期链路验证：仅平移初始位置，平地轨迹几乎相同'
            warning='<p>早期开发记录，未限制heightmap越界；不作为正式粗糙地形证据。</p>' if directory.name=='rough_transfer30m_eval' else ''
            evaluations.append(f'<h3>{name}</h3>{warning}<p>{data["tracking_successes"]}/8 回合达标，{data["falls"]}/8 跌倒，{data.get("out_of_bounds",0)}/8 越界；{perturbation}。协议：{data.get("protocol","development")}。<a href="{name}/result/evaluation.json">原始评估</a></p><div class="scroll"><table><tr><th>场景</th><th>初始seed</th><th>步数</th><th>速度RMSE</th><th>偏航RMSE</th><th>状态</th><th>跟踪</th></tr>{rows}</table></div>{plot}<div class="videos">{videos}</div>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A1 GPU 强化学习实验</title><style>body{background:#0c1420;color:#dee9f4;font:16px/1.7 system-ui;margin:0}main{max-width:1100px;margin:auto;padding:36px 20px}h1{font-size:34px}h2{border-top:1px solid #344254;padding-top:24px}a{color:#76ccfa}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:10px;border-bottom:1px solid #344254;text-align:left}.scroll{overflow:auto}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}video,img{width:100%;border-radius:12px}figure{margin:0}p{color:#b9cbda}</style><main><p>第五轮开发 / Unitree A1 / MuJoCo Playground × Brax PPO</p><h1>从 GPU 并行仿真到多指令策略评估</h1><p>复用固定开源快照的任务逻辑并适配 A1，实际运行 RTX 4060 Laptop。48维actor包含线速度，123维critic包含额外信息；本条路线尚未加入RMA历史适应器。所有新增环境、缓存与结果位于D盘。</p><h2>硬件与吞吐实测</h2><p>吞吐为同步后的物理环境控制步，不代表PPO完整训练吞吐。首次编译单独计时；后续运行复用Warp缓存。显存是5秒采样的整卡已用显存，包含桌面程序。</p><div class="scroll"><table><tr><th>试跑</th><th>JAX</th><th>环境数</th><th>控制步/秒</th><th>reset / step编译秒</th><th>显存峰值MiB</th></tr>'''+''.join(probes)+'''</table></div><p><a href="environment.json">真实设备与版本</a> · <a href="requirements-frozen.txt">冻结依赖</a></p><h2>已完成的训练</h2>'''+(''.join(trains) or '<p>尚无完成记录。</p>')+'''<h2>独立固定指令评估</h2><p>四类指令各2回合，每回合最长20秒，无自动重置。成功条件：完整存活、平面速度向量RMSE&lt;0.2 m/s、偏航RMSE&lt;0.3 rad/s。视频在首次跌倒处截断。单训练seed开发结果，不能说明跨seed稳定性；当前不代表台阶、坡地或真机通过。</p>'''+(''.join(evaluations) or '<p>尚无完成评估。</p>')+'</main></html>'
    summary_rows=[]
    for trial in preferred:
        data_path=root/trial/'result/evaluation.json'
        if not data_path.exists():continue
        data=json.loads(data_path.read_text())
        summary_rows.append(f'<tr><td>{trial}</td><td>{data["tracking_successes"]}/8</td><td>{data["falls"]}/8</td><td>{data.get("out_of_bounds",0)}/8</td></tr>')
    overview='<h2>当前验收与失败边界</h2><p>平地固定保留集8/8；粗糙地形中央起点续训后6/8，其中两个快速回合到达地图边界。更换空间起点后初次续训与直接迁移均4/8，说明局部成绩不能代表整张地形泛化。随后扩大训练起点分布，所有追加结果逐项列在下表。奖励调整与追加预算同时变化，不构成等预算单变量消融。</p><div class="scroll"><table><tr><th>固定测试 / 模型</th><th>达标</th><th>跌倒</th><th>越界</th></tr>'+''.join(summary_rows)+'</table></div><p>仅单训练种子，每组8回合；GPU路线无RMA历史适应器。台阶、坡地及真机尚未通过验证。粗糙图安全边界为|x|、|y|≤9.5m，新空间协议起点x=-2m，判据保持不变。</p>'
    if (root/'rough_wider_spatial_holdout/result/evaluation.json').exists():
        overview=overview.replace('所有追加结果逐项列在下表。','新空间保留集由直接迁移1/8变为宽起点续训7/8，跌倒由7/8变为1/8；旧确认集8/8。仍有快速回合跌倒。追加采样、起点分布和训练越界终止同时变化，不能单独归因；所有结果逐项列在下表。')
    page=page.replace('<h2>硬件与吞吐实测</h2>',overview+'<h2>硬件与吞吐实测</h2>')
    if (root/'training.png').exists():
        page=page.replace('<h2>独立固定指令评估</h2>','<h2>训练回调诊断</h2><p>每分支仅3个回调点；横轴为各分支新增步数，任务和起点分布不同。回报不是独立跟踪成功率，不跨任务比较绝对值。</p><img src="training.png" alt="训练回报及存活长度"><h2>独立固定指令评估</h2>')
    demos=[]
    for directory in sorted((root/'showcase').glob('*')):
        video=directory/'A1_terrain_demo.mp4';manifest=directory/'demo_manifest.json'
        if not video.exists() or not manifest.exists():continue
        record=json.loads(manifest.read_text())
        relative=directory.relative_to(root).as_posix()
        label='地形专门续训模型' if 'trained' in directory.name else '已有粗糙地形模型直接迁移'
        demos.append(f'<h3>{label}</h3><p>六级3.5cm台阶、6.84°上下坡、变速停走和转向。底层PPO速度策略＋高层航向反馈；固定保留集{record["aggregate_success"]}/8达标、{record["aggregate_falls"]}/8跌倒。视频选取每场景首个达标回合，保留完整20秒，1×真实时间。</p><video controls preload="metadata" poster="{relative}/poster.jpg" src="{relative}/A1_terrain_demo.mp4"></video><p><a href="{relative}/A1_terrain_demo.mp4" download>下载90秒展示视频</a> · <a href="{relative}/demo_manifest.json">片段清单及哈希</a></p>')
    if demos:
        page=page.replace('<h2>当前验收与失败边界</h2>','<h2>坡地与台阶：完整任务展示</h2>'+''.join(demos)+'<p>复用MuJoCo Playground任务与Brax PPO；航向指令接口参考legged_gym。本地几何为真实碰撞箱体。高层反馈和策略学习的作用分别记录；不是完整论文复现或真机演示。</p><h2>当前验收与失败边界</h2>')
        page=page.replace('当前不代表台阶、坡地或真机通过。','粗糙地形与后续台阶/坡地分别验收；所有结果均为仿真，无真机验证。')
        page=page.replace('台阶、坡地及真机尚未通过验证。','后续低台阶/缓坡的PPO＋高层航向反馈保留测试见上方视频；更高障碍和真机未验证。')
        page=page.replace('<h1>从 GPU 并行仿真到多指令策略评估</h1>','<h1>A1 从多指令运动到坡地与台阶穿越</h1><p><a href="showcase/index.html">打开简洁视频展示页 →</a></p>')
    import shutil
    cleanup=root.parent/'cleanup.json'
    if cleanup.exists():shutil.copy2(cleanup,root/'cleanup.json')
    page=page.replace('</main></html>','<h2>存储与无效实验</h2><p>已删除本任务旧环境/缓存3,070,421,821字节；C盘实际空闲约65.2GiB。新增环境、训练与视频保存在D盘。<a href="cleanup.json">清理记录</a>。未压缩或关闭用户原WSL环境。</p><ul>'+''.join(failures)+'</ul></main></html>')
    (root/'index.html').write_text(page,encoding='utf-8')
    print(root/'index.html')

if __name__=='__main__':main()
