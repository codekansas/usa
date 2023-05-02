import argparse

import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="out/point_cloud.ply")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.path)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
