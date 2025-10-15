import genesis as gs
import os

from pathlib import Path
current_file_path = Path(__file__).resolve().parent
root_path = str(current_file_path.parent.parent)
asset_path = os.path.join(root_path, 'src', 'assets')  # Use os.path.join for cross-platform compatibility

FRANKA_URDF = {
    "morph": gs.morphs.URDF(
        file=os.path.join("urdf", "panda_bullet", "panda.urdf"),
        pos=(-0.3, 0.0, 0.0),
        euler=(0, 0, 0),
        merge_fixed_links=False, 
        fixed=True,
    ),
    "material": gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
}

FRANKA_CONFIG = {
    "initial_dofs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04],
    "control": {
        "kp": [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
        "kv": [450, 450, 350, 350, 200, 200, 200, 10, 10],
        "force_range_min": [-87, -87, -87, -87, -12, -12, -12, -100, -100],
        "force_range_max": [87, 87, 87, 87, 12, 12, 12, 100, 100],
    }
}

FRANKA_MJCF = {
    "morph": gs.morphs.MJCF(
        file=os.path.join("xml", "franka_emika_panda", "panda.xml"),
    ),
    "material": gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
    "vis_mode": "collision",
}

SATELLITE = {
    "morph": gs.morphs.URDF(
        file=os.path.join(asset_path, 'urdf', 'satellite', 'urdf', 'satellite.urdf'),
        scale=5e-1,
        merge_fixed_links=False, 
        fixed=False,
    ),
    "material": gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
}

SATELLITE_PART = {
    "morph": gs.morphs.URDF(
        file=os.path.join(asset_path, 'urdf', 'satellite_battery', 'urdf', 'satellite_battery.urdf'),
        pos=(-1.1, 0.3, 1.0),
        euler=(0.0, 90.0, 0.0),
        scale=5e-1,
        merge_fixed_links=False,
        fixed=False,
    ),
    "material": gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
}

ASSETS = {
    "franka": FRANKA_URDF,
    "satellite": SATELLITE,
    "satellite_part": SATELLITE_PART,
}