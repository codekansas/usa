from pathlib import Path

from usa.tasks.datasets.pybullet import PyBulletDataset

ds_path = Path(__file__).parent.resolve() / "data" / "04_recorded_clip.pkl"
ds = PyBulletDataset(ds_path)

print(ds.poses[..., :3, 3])
