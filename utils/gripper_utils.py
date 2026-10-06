# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities to represent an end-effector."""

import os

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn
import transforms3d
import trimesh as tm
import trimesh.sample
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere

from utils import math_utils


class HandModel:
    """Hand model class based on: https://github.com/tengyu-liu/GenDexGrasp/blob/main/utils_model/HandModel.py."""

    def __init__(
        self,
        robot_name,
        urdf_filename,
        mesh_path,
        batch_size=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hand_scale=2.0,
        data_dir="data",
    ):
        self.device = device
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(
            dtype=torch.float, device=self.device
        )
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        if robot_name == "allegro_right":
            self.robot_name = "allegro_right"
            robot_name = "allegro"
        else:
            self.robot_name = robot_name

        self.global_translation = None
        self.global_rotation = None

        self.contact_point_basis = {}
        self.contact_normals = {}
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.key_point_idx_dict = {}
        self.mesh_verts = {}
        self.mesh_faces = {}

        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []

        for link in visual.links:
            if not link.visuals:
                continue
            if isinstance(link.visuals[0].geometry, Mesh):
                if robot_name == "shadowhand" or robot_name == "allegro" or robot_name == "barrett":
                    filename = link.visuals[0].geometry.filename.split("/")[-1]
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force="mesh", process=False)
                if robot_name == "shadowhand":
                    T = trimesh.transformations.rotation_matrix(
                        np.radians(-90), [1, 0, 0]
                    )  # rotate -90° around X-axis
                    mesh.apply_transform(T)
            elif isinstance(link.visuals[0].geometry, Cylinder):
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius,
                    height=link.visuals[0].geometry.length,
                )
            elif isinstance(link.visuals[0].geometry, Box):
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif isinstance(link.visuals[0].geometry, Sphere):
                mesh = tm.primitives.Sphere(radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(link.visuals[0].geometry.scale).reshape([1, 3])
            except Exception:  # pylint: disable=broad-exception-caught
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])

            except Exception:  # pylint: disable=broad-exception-caught
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            count = 512

            pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=count)
            pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            pts *= scale
            if robot_name == "shadowhand":
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = (
                torch.from_numpy(pts).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            )
            self.surface_points_normal[link.name] = (
                torch.from_numpy(pts_normal)
                .to(device)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == "shadowhand":
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = (
                np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            )
            self.mesh_faces[link.name] = np.array(mesh.faces)

        self.scale = hand_scale

        # new 2.1
        self.revolute_joints = []
        for i, _ in enumerate(self.robot_full.joints):
            if self.robot_full.joints[i].joint_type in ["revolute", "prismatic"]:
                self.revolute_joints.append(self.robot_full.joints[i])
        self.revolute_joints_q_upper = []
        self.revolute_joints_q_lower = []
        for i, _ in enumerate(self.robot.get_joint_parameter_names()):
            for j, _ in enumerate(self.revolute_joints):
                if self.revolute_joints[j].name == self.robot.get_joint_parameter_names()[i]:
                    joint = self.revolute_joints[j]
                    assert joint.name == self.robot.get_joint_parameter_names()[i]
                    self.revolute_joints_q_lower.append(joint.limit.lower)
                    self.revolute_joints_q_upper.append(joint.limit.upper)

        joint_lower = np.array(self.revolute_joints_q_lower)
        joint_upper = np.array(self.revolute_joints_q_upper)
        joint_mid = (joint_lower + joint_upper) / 2
        joints_q = (joint_mid + joint_lower) / 2
        self.rest_pose = (
            torch.from_numpy(np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q]))
            .unsqueeze(0)
            .to(device)
            .float()
        )

        self.revolute_joints_q_lower = (
            torch.Tensor(self.revolute_joints_q_lower).repeat([self.batch_size, 1]).to(device)
        )
        self.revolute_joints_q_upper = (
            torch.Tensor(self.revolute_joints_q_upper).repeat([self.batch_size, 1]).to(device)
        )

        self.rest_pose = self.rest_pose.repeat([self.batch_size, 1])

        self.current_status = None

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]

        self.global_rotation = math_utils.robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def to(self, device):
        """Move all tensors to the specified device for DataParallel compatibility."""
        if self.device == device:
            return self

        self.device = device

        # Move pytorch_kinematics chain
        self.robot = self.robot.to(device=device)

        # Move surface points and normals
        for link_name in self.surface_points:
            self.surface_points[link_name] = self.surface_points[link_name].to(device)
        for link_name in self.surface_points_normal:
            self.surface_points_normal[link_name] = self.surface_points_normal[link_name].to(device)

        # Move joint limits and rest pose
        self.revolute_joints_q_lower = self.revolute_joints_q_lower.to(device)
        self.revolute_joints_q_upper = self.revolute_joints_q_upper.to(device)
        self.rest_pose = self.rest_pose.to(device)

        # Move contact point basis and normals if they exist
        for link_name in self.contact_point_basis:
            self.contact_point_basis[link_name] = self.contact_point_basis[link_name].to(device)
        for link_name in self.contact_normals:
            self.contact_normals[link_name] = self.contact_normals[link_name].to(device)

        return self

    def get_surface_points(self, q=None, downsample=False, num_points=1024):
        """Returns surface points on the end-effector on a given pose.

        With `downsample=True` a random subset of `num_points` points is
        returned. The default matches `model.robot_pc_n` in the Hydra config;
        callers without config access (the visualization scripts) rely on it,
        so change both together.
        """

        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            if self.robot_name == "robotiq_3finger" and link_name == "gripper_palm":
                continue
            if self.robot_name == "robotiq_3finger_real_robot" and link_name == "palm":
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        if downsample:
            surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][
                :, :num_points
            ]
        return surface_points * self.scale

    def get_key_points_from_indices(self, key_point_idx_dict, q=None):
        """Returns keypoints from a set of indices when in a given pose."""

        if q is not None:
            self.update_kinematics(q)

        key_points = []
        for link_name in key_point_idx_dict:
            trans_matrix = self.current_status[link_name].get_matrix()
            pts = np.concatenate(
                [
                    self.mesh_verts[link_name],
                    np.ones([len(self.mesh_verts[link_name]), 1]),
                ],
                axis=-1,
            )
            surface_points = torch.from_numpy(pts).float().unsqueeze(0).repeat(1, 1, 1)

            key_point_idx = key_point_idx_dict[link_name]
            key_points.append(
                torch.matmul(
                    trans_matrix,
                    surface_points[:, key_point_idx.long(), :].transpose(1, 2),
                ).transpose(1, 2)
            )

        key_points = torch.cat(key_points, dim=1)[..., :3]
        key_points = torch.matmul(self.global_rotation, key_points.transpose(1, 2)).transpose(
            1, 2
        ) + self.global_translation.unsqueeze(1)
        return (key_points * self.scale).squeeze(0)

    def get_meshes_from_q(self, q=None, i=0):
        """Returns gripper meshes in a given pose."""

        data = []
        if q is not None:
            self.update_kinematics(q)
        for _, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color="lightblue", opacity=1.0):
        """Returns plot data for the gripper in a given pose."""

        data = []
        if q is not None:
            self.update_kinematics(q)
        for _, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(
                    x=transformed_v[:, 0],
                    y=transformed_v[:, 1],
                    z=transformed_v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                )
            )
        return data
