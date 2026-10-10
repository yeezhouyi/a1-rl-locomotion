"""Build A1 sensor-equipped models for the upstream Playground joystick task."""
from pathlib import Path
import argparse
import json
import copy
import hashlib
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source = ROOT / "assets/unitree_a1"
    shutil.copytree(source / "assets", destination / "assets")
    root = ET.parse(source / "a1.xml").getroot()
    option = root.find("option")
    option.attrib.update(timestep="0.004", iterations="4", ls_iterations="5", integrator="Euler", cone="pyramidal", impratio="1")
    root.find("default/default/joint").set("frictionloss", "0.0")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.1", contype="1", conaffinity="0", friction="0.6", condim="3", rgba="0.45 0.5 0.55 1")
    trunk = world.find("body[@name='trunk']")
    ET.SubElement(trunk, "site", name="imu", pos="0 0 0", size="0.008", group="5")
    ET.SubElement(trunk, "camera", name="track", pos="1 -2 0.8", xyaxes="0.9 0.4 0 -0.2 0.5 0.85", mode="trackcom")
    sensor = ET.SubElement(root, "sensor")
    for tag, name in [("gyro", "gyro"), ("velocimeter", "local_linvel"), ("accelerometer", "accelerometer")]:
        ET.SubElement(sensor, tag, site="imu", name=name)
    for tag, name in [("framezaxis", "upvector"), ("framelinvel", "global_linvel"), ("frameangvel", "global_angvel")]:
        ET.SubElement(sensor, tag, objtype="site", objname="imu", name=name)
    # Every primitive collider touches ground; self collisions are disabled.
    # Keep trunk/thigh collisions: unlike the upstream feet-only model,
    # the A1 cannot pass its non-foot body through the ground.
    for geom in world.iter("geom"):
        if geom.get("name") == "floor":
            continue
        visual = geom.get("class") == "visual"
        geom.attrib.update(contype="0", conaffinity="0" if visual else "1", condim="3")
    for leg in ["FR", "FL", "RR", "RL"]:
        calf = world.find(f".//body[@name='{leg}_calf']")
        calf.find("geom[@class='foot']").set("name", leg)
        ET.SubElement(calf, "site", name=leg, pos="0 0 -0.2", size="0.008", group="5")
        ET.SubElement(sensor, "framelinvel", objtype="site", objname=leg, name=f"{leg}_global_linvel")
        ET.SubElement(sensor, "framepos", objtype="site", objname=leg, name=f"{leg}_pos", reftype="site", refname="imu")
        ET.SubElement(sensor, "contact", name=f"{leg}_floor_found", geom1=leg, geom2="floor", reduce="mindist", num="1", data="found")
    ET.indent(root)
    ET.ElementTree(root).write(destination / "a1_flat.xml", encoding="unicode")
    # Directly reuse the fixed upstream rough-terrain heightmap (Apache-2.0),
    # retaining its size/scale and footprint. No rocky texture is imported.
    rough = copy.deepcopy(root)
    upstream_heightmap = ROOT / "outputs/opensource_reference/mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/assets/hfield.png"
    shutil.copy2(upstream_heightmap, destination / "assets/hfield.png")
    ET.SubElement(rough.find("asset"), "hfield", name="hfield", file="hfield.png", size="10 10 .05 1.0")
    floor = rough.find("worldbody/geom[@name='floor']")
    floor.attrib.pop("size")
    floor.attrib.update(type="hfield", hfield="hfield", friction="1.0")
    key = rough.find("keyframe/key[@name='home']")
    qpos = key.get("qpos").split(); qpos[2] = "0.35"; key.set("qpos", " ".join(qpos))
    ET.ElementTree(rough).write(destination / "a1_rough.xml", encoding="unicode")
    (destination / "heightmap_source.json").write_text(json.dumps(dict(
        repo="https://github.com/google-deepmind/mujoco_playground", commit="d59156e099a8785dd58fbce2e25a6ee1f7b5800c",
        path=upstream_heightmap.relative_to(ROOT / "outputs/opensource_reference/mujoco_playground").as_posix(),
        sha256=hashlib.sha256(upstream_heightmap.read_bytes()).hexdigest(),license="Apache-2.0"),indent=2))
    for file in source.glob("*LICENSE*"):
        shutil.copy2(file, destination / file.name)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination / "a1_flat.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    build(parser.parse_args().out)
