"""Helpers shared by the cluttered scene scripts in this folder.

Functions that touch Isaac Sim import omni/pxr lazily, so this module can be imported
before the SimulationApp is launched.
"""
import os
import re
import sys
import json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


######################### Simulation #########################
def create_world(physics_dt=1/240, gravity=-9.81):
    from omni.isaac.core import World
    world = World(physics_dt=physics_dt, set_defaults=True)
    world.reset()
    physicsContext = world.get_physics_context()
    physicsContext.set_solver_type("TGS")
    physicsContext.enable_gpu_dynamics(True)
    physicsContext.enable_stablization(True)
    physicsContext.set_gravity(gravity)
    world.reset()
    return world

def debug_pause(world, step=True, num_steps=1000000000000, render=True):
    """Keep rendering the paused world for num_steps (also used to let the viewport settle before screenshots)."""
    if step: world.step(render=render)
    world.pause()
    for _ in range(num_steps):
        world.step(render=render)

def import_gripper_with_physics(world, work_path, usd_path, EF_axis):
    """Spawn a gripper with self collisions enabled, 1 m above the workstation so it does not
    start inside the ground plane or the scene objects."""
    from standalone import import_gripper
    from omni.isaac.core.articulations import ArticulationView
    gripper_art = import_gripper(world, work_path, usd_path, EF_axis, self_collision=True, z_offset=1.0)
    gripper_view = world.scene.add(
        ArticulationView(
            prim_paths_expr=work_path,
            reset_xform_properties=False
        )
    )
    return gripper_art, gripper_view

def get_matching_prim_paths(world, path_expr):
    pattern = re.compile(rf"{path_expr}")
    return [str(prim.GetPath()) for prim in world.stage.Traverse() if pattern.match(str(prim.GetPath()))]


######################### Grasps #########################
def load_grasps(grasp_dir, gripper_name, object_name, grasp_file=""):
    """Load <grasp_dir>/<gripper>/<gripper>-<object>.json (or <grasp_dir>/<grasp_file> if given)."""
    if grasp_file == "":
        grasp_file = os.path.join(grasp_dir, f"{gripper_name}/{gripper_name}-{object_name}.json")
    else:
        grasp_file = os.path.join(grasp_dir, grasp_file)
    if not os.path.exists(grasp_file):
        print(f"No grasp file found at {grasp_file}. Skipping...")
        return {"pose": [], "dofs": [], "stable_poses": [], "stable_dofs": []}
    with open(grasp_file, "r") as f:
        return json.load(f)

def calc_approach_angle(gripper_trans, obj_trans):
    """Elevation angle (rad) of the gripper origin as seen from the object. Batched over rows."""
    diff_vec = gripper_trans - obj_trans
    horizontal_dist = np.linalg.norm(diff_vec[:, :2], axis=1)
    vertical_dist = diff_vec[:, 2]
    return np.arctan2(vertical_dist, horizontal_dist)

def calc_approach_angle_bot(gripper_trans, obj_trans):
    """Approach angle measured from the object's projection on the ground, so grasps on the lower half
    of an object are not all rejected."""
    obj_trans_zeroes = obj_trans.copy()
    obj_trans_zeroes[:, 2] = 0.0
    return calc_approach_angle(gripper_trans, obj_trans_zeroes)


######################### Meshes / point clouds #########################
def get_mesh_vertices(world, mesh_path, world_tf_matrix):
    """Return (vertices in world frame, face vertex counts, face vertex indices) of a UsdGeom.Mesh."""
    from pxr import UsdGeom
    from omni.isaac.core.utils.transformations import pose_from_tf_matrix, tf_matrix_from_pose

    mesh_prim = world.stage.GetPrimAtPath(mesh_path)
    if not mesh_prim.IsValid():
        return None, None, None
    mesh = UsdGeom.Mesh(mesh_prim)
    vertices = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

    # Transform vertices to world space
    vertices = np.array([tf_matrix_from_pose(np.array(v), np.array([0, 0, 0, 1])) for v in vertices])

    mesh_tf_matrix = mesh.GetLocalTransformation()
    mesh_tf_matrix = np.array(mesh_tf_matrix).transpose()

    vertices = np.matmul(mesh_tf_matrix, vertices)
    vertices = np.matmul(world_tf_matrix, vertices)
    vertices = np.array([pose_from_tf_matrix(v)[0][:3] for v in vertices])

    return vertices, np.array(face_vertex_counts), np.array(face_vertex_indices)

def face_sampling_from_mesh(vertices, face_vertex_counts, face_vertex_indices, num_samples=10000, weight_by_area=True):
    """Sample points uniformly on the surface of a mesh (non-triangular faces are skipped)."""
    face_idx = 0
    faces = []
    face_areas = []
    for count in face_vertex_counts:
        if count != 3:
            face_idx += count
            continue
        idx0, idx1, idx2 = face_vertex_indices[face_idx:face_idx+3]
        v0, v1, v2 = np.array(vertices[idx0]), np.array(vertices[idx1]), np.array(vertices[idx2])
        faces.append((v0, v1, v2))
        face_areas.append(np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0)
        face_idx += 3

    # Weigh faces by area to sample surface uniformly
    face_weights = None
    if weight_by_area:
        face_areas = np.array(face_areas)
        face_weights = face_areas / face_areas.sum()

    # Barycentric sampling: https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    sampled_points = []
    for face in np.random.choice(len(faces), size=num_samples, p=face_weights, replace=True):
        v0, v1, v2 = faces[face]
        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        sampled_points.append((1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + (sqrt_r1 * r2) * v2)
    return np.array(sampled_points)

def farthest_point_sampling(points, num_samples):
    if num_samples >= len(points):
        return points

    sampled_indices = np.zeros(num_samples, dtype=int)
    distances = np.full(len(points), np.inf)

    sampled_indices[0] = np.random.randint(len(points))
    for i in range(1, num_samples):
        dists = np.linalg.norm(points - points[sampled_indices[i-1]], axis=1)
        distances = np.minimum(distances, dists)
        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices]

def add_noise_to_points(points, noise_std):
    return points + np.random.normal(0, noise_std, size=points.shape)

def save_point_cloud_to_ply(points, output_path):
    from plyfile import PlyData, PlyElement
    ply_vertices = np.array([(p[0], p[1], p[2]) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([PlyElement.describe(ply_vertices, "vertex")]).write(output_path)
    print(f"Point cloud saved to {output_path}")
