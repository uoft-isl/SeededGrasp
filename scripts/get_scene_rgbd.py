"""Render RGB images and point clouds of generated scenes from five fixed cameras.

For every scene folder in --scene_dir (output of gen_clutter_scene.py) this writes
<camera>_rgb.png and <camera>_depth.pcd into the scene folder and adds the camera poses to <scene>.json.
"""
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CameraWrapper:
    def __init__(self, name, position, orientation, res=640):
        from omni.isaac.sensor import Camera
        self.name = name
        self.position = position
        self.orientation = orientation

        self.res = res
        self.noise = 0.01

        self.camera = Camera(
            prim_path=f"/World/CameraArray/{name}",
            position=position,
            orientation=orientation,
            frequency=1,
            resolution=(self.res, self.res),
        )

    def initialize(self):
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()

    def crop_rgbd_image(self, rgb_image, pcd, crop_factor=0.7):
        h, w = rgb_image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2

        rgb_image_cropped = rgb_image[start_h:start_h + new_h, start_w:start_w + new_w]
        pcd_cropped = pcd.reshape(rgb_image.shape[0], rgb_image.shape[1], 3)[start_h:start_h + new_h, start_w:start_w + new_w].reshape(-1, 3)

        return rgb_image_cropped, pcd_cropped
    
    def add_noise_to_pcd(self, pcd):
        noise = np.random.normal(0, self.noise, pcd.shape)
        pcd_noisy = pcd + noise
        return pcd_noisy

    def get_rgbd_image(self):
        rgb_image = self.camera.get_rgb()
        depth_image = self.camera.get_pointcloud()

        rgb_image, depth_image = self.crop_rgbd_image(rgb_image, depth_image)
        #depth_image = self.add_noise_to_pcd(depth_image)

        rgb_image = (rgb_image).astype(np.uint8)
        rgb_image = o3d.geometry.Image(rgb_image)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_image)

        return rgb_image, pcd
    
    def save_rgbd_image(self, save_path, rgb_image, pcd):
        rgb_path = os.path.join(save_path, f"{self.name}_rgb.png")
        depth_path = os.path.join(save_path, f"{self.name}_depth.pcd")

        o3d.io.write_image(rgb_path, rgb_image)
        o3d.io.write_point_cloud(depth_path, pcd)

        print(f"Saved RGB image to {rgb_path}"
              f" and depth point cloud to {depth_path}")

    def get_world_pose(self):
        return self.camera.get_world_pose()
        
    def save_camera_pose(self, save_path):
        pose = self.camera.get_world_pose()
        scene_info = {}
        with open(f"{save_path}/{save_path.split('/')[-1]}.json", 'r') as f:
            scene_info = json.load(f)
        scene_info[self.name] = {
            "position": pose[0].tolist(),
            "orientation": pose[1].tolist()
        }
        with open(f"{save_path}/{save_path.split('/')[-1]}.json", 'w') as f:
            json.dump(scene_info, f, indent=4)

class CameraArray:
    def __init__(self):
        # Fixed bird's-eye view + four side views looking at the origin
        self.cameras = [
            CameraWrapper(name="camera_bev", position=np.array([0, 0, 3.25]), orientation=np.array([0, -0.707, 0, 0.707])),  # wxyz
            CameraWrapper(name="camera_north", position=np.array([2.5, 0, 1.75]), orientation=np.array([0, -0.301, 0, 0.954])),  # wxyz
            CameraWrapper(name="camera_south", position=np.array([-2.5, 0, 1.75]), orientation=np.array([0.954, 0, 0.301, 0])),  # wxyz
            CameraWrapper(name="camera_east", position=np.array([0, -2.5, 1.75]), orientation=np.array([0.674, -0.213, 0.213, 0.674])),  # wxyz
            CameraWrapper(name="camera_west", position=np.array([0, 2.5, 1.75]), orientation=np.array([-0.674, -0.213, -0.213, 0.674])),  # wxyz
        ]

    def initialize(self):
        for camera in self.cameras:
            camera.initialize()

    def get_rgbd_image(self):
        rgbd_images = []
        pcds = []
        for camera in self.cameras:
            rgb_image, pcd = camera.get_rgbd_image()
            rgbd_images.append(rgb_image)
            pcds.append(pcd)
        return rgbd_images, pcds

    def get_world_pose(self):
        positions = []
        orientations = []
        for camera in self.cameras:
            pose = camera.get_world_pose()
            positions.append(pose[0])
            orientations.append(pose[1])
        return [positions, orientations]

    def save_rgbd_images(self, save_path, rgbd_images, pcds):
        for i, camera in enumerate(self.cameras):
            camera.save_rgbd_image(save_path, rgbd_images[i], pcds[i])

    def save_camera_pose(self, save_path):
        for camera in self.cameras:
            camera.save_camera_pose(save_path)

def make_ground_green(world):
    from pxr import UsdShade
    from omni.isaac.core.materials import PreviewSurface
    ground_plane = world.stage.GetPrimAtPath("/World/defaultGroundPlane")
    material_binding_api = UsdShade.MaterialBindingAPI(ground_plane)
    material_binding_api.UnbindAllBindings()
    ground_material = PreviewSurface(
        prim_path="/World/defaultGroundPlane/Looks/ground_material",
        color=np.array([0.36, 0.7, 0.25]),
    )
    material_binding_api.Bind(
        ground_material.material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RGBD images from a scene in Isaac Sim")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene USD file and other assets")
    parser.add_argument("--device", type=int, default=0, help="GPU device index to use for simulation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Check for scenes
    if not os.path.exists(args.scene_dir):
        raise FileNotFoundError(f"Scene directory {args.scene_dir} does not exist.")
    for dir in os.listdir(args.scene_dir):
        if not os.path.isdir(os.path.join(args.scene_dir, dir)):
            continue
        scene_path = os.path.join(args.scene_dir, dir)
        usd_file = os.path.join(scene_path, f"{dir}.usd")
        if not os.path.exists(usd_file):
            raise FileNotFoundError(f"USD file {usd_file} does not exist in {scene_path}")
        if not os.path.exists(os.path.join(scene_path, f"{dir}.json")):
            raise FileNotFoundError(f"JSON file {scene_path}/{dir}.json does not exist in {scene_path}")

    ######################### Launch Isaac Sim #########################
    from omni.isaac.kit import SimulationApp
    config = {
        "headless": True,
        "max_bounces": 0,
        "fast_shutdown": True,
        "max_specular_transmission_bounces": 0,
        "physics_gpu": args.device,
        "active_gpu": args.device,
        "multi_gpu": True
    }
    simulation_app = SimulationApp(config)

    # Omniverse imports
    import omni.usd
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core.utils.stage import is_stage_loading

    # Custom imports
    from sim_utils import add_light
    from scene_utils import create_world

    # Expect scenes to be provided in the form of directories each containing .usd, .json, and any pngs/pcds
    # Iterate through each scene directory
    for dir in tqdm(os.listdir(args.scene_dir)):
        if not os.path.isdir(os.path.join(args.scene_dir, dir)):
            continue
        scene_path = os.path.join(args.scene_dir, dir)
        print(f"Processing scene: {scene_path}")
        
        # Load scene provided in USD
        omni.usd.get_context().open_stage(os.path.join(scene_path, f"{dir}.usd"))
        simulation_app.update()
        simulation_app.update()
        while is_stage_loading():
            simulation_app.update()
        world = create_world(gravity=-9.8)
        world.reset()

        # Setup scene (put scene augmentations here)
        # make_ground_green(world)
        camera_array = CameraArray()
        world.reset()
        camera_array.initialize()
        add_light()

        # Take RGBD images
        for _ in range(100): # Need to render for certain #steps for stable images
            world.step(render=True)
        world.pause()

        rgbd_images, pcds = camera_array.get_rgbd_image()

        # Perform RGBD augmentations if needed

        # Save RGBD images to the scene directory
        camera_array.save_rgbd_images(scene_path, rgbd_images, pcds)

        # Save camera poses to the .json file
        camera_array.save_camera_pose(scene_path)

        # Clean up
        stage_utils.create_new_stage()

    simulation_app.close()
    print("All scenes processed and RGBD images saved.")
