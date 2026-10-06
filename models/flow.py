# Copyright 2026 The SeededGrasp Authors
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

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from scipy.spatial.transform import Rotation as R

from models.gnn import GCN
from models.mlp import MLP
from utils import math_utils
from utils.general_utils import get_handmodel
from utils.gnn_utils import generate_adj_mat_feats
from utils.normalization import normalize_pc, unnormalize_q
from utils_data.rgbd_scene_dataset import (
    apply_gripper_rotation_alignment_with_transforms,
    build_gripper_alignment_transforms,
)

device = torch.device("cuda")

"""
https://github.com/eje24/iap-diffusion-labs/blob/main/solutions/lab_three_complete.ipynb
"""


def compute_gradient_and_curvature(points, adj, epsilon=1e-8):
    """
    Computes normals (gradient) and curvature (surface variation) using PCA
    on the neighborhoods defined by the adjacency matrix.

    Args:
        points: (bs, N, 3) Tensor containing point coordinates.
        adj: (bs, N, N) Adjacency matrix (0/1 or weights), can be sparse or dense.
             Can also be a list of sparse tensors.
             Should define the k-neighbors (including self-loops is recommended).
        epsilon: Small scalar to prevent division by zero.

    Returns:
        normals: (bs, N, 3) The estimated surface normals.
        curvature: (bs, N) The estimated surface variation (0 to 1/3).
    """
    bs, N, _ = points.shape

    # Handle list of sparse tensors
    if isinstance(adj, list):
        # Process each sample individually with sparse operations
        all_normals = []
        all_curvature = []
        for b in range(bs):
            adj_b = adj[b]  # sparse (N, N)
            pts_b = points[b]  # (N, 3)
            normals_b, curvature_b = _compute_gradient_and_curvature_single_sparse(
                pts_b, adj_b, epsilon
            )
            all_normals.append(normals_b)
            all_curvature.append(curvature_b)
        return torch.stack(all_normals, dim=0), torch.stack(all_curvature, dim=0)

    # Check if adj is a batched sparse tensor or dense
    if adj.is_sparse:
        # For batched sparse, we need to process individually
        all_normals = []
        all_curvature = []
        for b in range(bs):
            # Extract the b-th sparse matrix (this is tricky for batched sparse)
            # Assuming adj is a list or we convert
            adj_b = adj[b] if adj.dim() == 3 else adj
            pts_b = points[b]
            normals_b, curvature_b = _compute_gradient_and_curvature_single_sparse(
                pts_b, adj_b, epsilon
            )
            all_normals.append(normals_b)
            all_curvature.append(curvature_b)
        return torch.stack(all_normals, dim=0), torch.stack(all_curvature, dim=0)

    # Dense path (original implementation)
    # 1. Compute Degree (k) for each point
    # Shape: (bs, N, 1)
    k = torch.sum(adj, dim=2, keepdim=True).clamp(min=1)

    # 2. Compute Centroids (E[X])
    # The centroid of neighbors for each point i.
    # (bs, N, N) @ (bs, N, 3) -> (bs, N, 3)
    centroids = torch.bmm(adj, points) / k

    # 3. Compute Covariance Matrix (E[XX^T] - E[X]E[X]^T)
    # x, y, z are (bs, N)
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # Helper to compute weighted sum of products over neighbors
    def weighted_sum(v1, v2):
        return torch.bmm(adj, (v1 * v2).unsqueeze(-1)).squeeze(-1)

    k_scalar = k.squeeze(-1)

    # E[XX^T] terms
    cov_xx = weighted_sum(x, x) / k_scalar
    cov_xy = weighted_sum(x, y) / k_scalar
    cov_xz = weighted_sum(x, z) / k_scalar
    cov_yy = weighted_sum(y, y) / k_scalar
    cov_yz = weighted_sum(y, z) / k_scalar
    cov_zz = weighted_sum(z, z) / k_scalar

    # Subtract (E[X])^2 terms: centroids * centroids^T
    c_x, c_y, c_z = centroids[..., 0], centroids[..., 1], centroids[..., 2]

    cov_xx = cov_xx - c_x * c_x
    cov_xy = cov_xy - c_x * c_y
    cov_xz = cov_xz - c_x * c_z
    cov_yy = cov_yy - c_y * c_y
    cov_yz = cov_yz - c_y * c_z
    cov_zz = cov_zz - c_z * c_z

    # 4. Construct the Covariance Tensor
    cov_matrix = torch.stack(
        [cov_xx, cov_xy, cov_xz, cov_xy, cov_yy, cov_yz, cov_xz, cov_yz, cov_zz], dim=-1
    ).reshape(bs, N, 3, 3)

    # 5. Eigen Decomposition using SVD (more stable with DataParallel)
    # For symmetric matrices, SVD gives same result as eigendecomposition
    # but is more thread-safe
    try:
        e_vals, e_vecs = torch.linalg.eigh(cov_matrix)
    except RuntimeError:
        # Fallback to SVD for DataParallel thread-safety issues
        U, S, Vh = torch.linalg.svd(cov_matrix)
        e_vals = S
        e_vecs = U

    # 6. Extract Normals (smallest eigenvalue)
    normals = e_vecs[..., 0]

    # 7. Extract Curvature
    sum_evals = torch.sum(e_vals, dim=-1).clamp(min=epsilon)
    curvature = e_vals[..., 0] / sum_evals

    return normals, curvature


def _compute_gradient_and_curvature_single_sparse(points, adj, epsilon=1e-8):
    """
    Compute gradient and curvature for a single sample with sparse adjacency matrix.

    Args:
        points: (N, 3) Tensor containing point coordinates.
        adj: (N, N) Sparse adjacency matrix.
        epsilon: Small scalar to prevent division by zero.

    Returns:
        normals: (N, 3) The estimated surface normals.
        curvature: (N,) The estimated surface variation.
    """
    N = points.shape[0]

    # Convert to dense for the matrix operations (on GPU this is fast)
    # The key savings is that we transfer sparse to GPU, then convert to dense on GPU
    adj_dense = adj.to_dense() if adj.is_sparse else adj

    # 1. Compute Degree (k) for each point
    k = torch.sum(adj_dense, dim=1, keepdim=True).clamp(min=1)  # (N, 1)

    # 2. Compute Centroids (E[X])
    centroids = torch.mm(adj_dense, points) / k  # (N, 3)

    # 3. Compute Covariance Matrix
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    def weighted_sum(v1, v2):
        return torch.mv(adj_dense, v1 * v2)

    k_scalar = k.squeeze(-1)

    cov_xx = weighted_sum(x, x) / k_scalar
    cov_xy = weighted_sum(x, y) / k_scalar
    cov_xz = weighted_sum(x, z) / k_scalar
    cov_yy = weighted_sum(y, y) / k_scalar
    cov_yz = weighted_sum(y, z) / k_scalar
    cov_zz = weighted_sum(z, z) / k_scalar

    c_x, c_y, c_z = centroids[:, 0], centroids[:, 1], centroids[:, 2]

    cov_xx = cov_xx - c_x * c_x
    cov_xy = cov_xy - c_x * c_y
    cov_xz = cov_xz - c_x * c_z
    cov_yy = cov_yy - c_y * c_y
    cov_yz = cov_yz - c_y * c_z
    cov_zz = cov_zz - c_z * c_z

    # 4. Construct the Covariance Tensor
    cov_matrix = torch.stack(
        [cov_xx, cov_xy, cov_xz, cov_xy, cov_yy, cov_yz, cov_xz, cov_yz, cov_zz], dim=-1
    ).reshape(N, 3, 3)

    # 5. Eigen Decomposition using SVD (more stable with DataParallel)
    try:
        e_vals, e_vecs = torch.linalg.eigh(cov_matrix)
    except RuntimeError:
        # Fallback to SVD for DataParallel thread-safety issues
        U, S, Vh = torch.linalg.svd(cov_matrix)
        e_vals = S
        e_vecs = U

    # 6. Extract Normals
    normals = e_vecs[..., 0]

    # 7. Extract Curvature
    sum_evals = torch.sum(e_vals, dim=-1).clamp(min=epsilon)
    curvature = e_vals[..., 0] / sum_evals

    return normals, curvature


class Scheduler(ABC):
    """
    Abstract base class for noise scheduler.
    """

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the noise schedule value for given time t.
        """
        pass

    @abstractmethod
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the derivative of the noise schedule with respect to time t.
        """
        pass


class LinearAlpha(Scheduler):
    """
    Linear noise scheduler for alpha_t.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns alpha_t for given time t.
        Args:
            - t: time (bs, 1)
        Returns:
            - alpha_t (bs, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns d/dt alpha_t for given time t.
        Args:
            - t: time (bs, 1)
        Returns:
            - d/dt alpha_t (bs, 1)
        """
        return torch.ones_like(t)


class LinearBeta(Scheduler):
    """
    Linear noise scheduler for beta_t.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns beta_t for given time t.
        Args:
            - t: time (bs, 1)
        Returns:
            - beta_t (bs, 1)
        """
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Returns d/dt beta_t for given time t.
        Args:
            - t: time (bs, 1)
        Returns:
            - d/dt beta_t (bs, 1)
        """
        return -torch.ones_like(t)


class ConditionalProbabilityPath(ABC):
    """
    Abstract base class for conditional probability paths.
    """

    @abstractmethod
    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, x0: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns samples from the distribution p_t(x|z).
        Args:
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - x: samples from dist (bs, p)
        """
        pass

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z).
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - u_t: conditional vector field (bs, p)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional score for p_t(x|z).
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - s_t: conditional score (bs, p)
        """
        pass


class R3GaussianCPP(ConditionalProbabilityPath):
    """
    Gaussian conditional probability path.
    """

    def __init__(self, cfg, alpha: Scheduler, beta: Scheduler):
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta

    def sample_x_init(self, z: torch.Tensor) -> torch.Tensor:
        """
        Samples the initial point x from the distribution p_0(x|z).
        Args:
            - z: gt value (bs, p)
        Returns:
            - x0: sampled initial point (bs, p)
        """

        return torch.fmod(torch.randn_like(z) * 0.3, 1.0)

    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, x0: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns samples from the Gaussian distribution p_t(x|z).
        Args:
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - x: samples from dist (bs, p)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)

        # Sample noise
        if x0 is None:
            noise = self.sample_x_init(z)
        else:
            noise = x0

        # Compute conditional path
        z_conditional = alpha_t * z + beta_t * noise

        return z_conditional, noise

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z) for a Gaussian probability path.
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - u_t: conditional vector field (bs, p)
        """
        # With the linear schedule (alpha_t = t, beta_t = 1 - t) the conditional
        # vector field of the Gaussian path reduces to the straight-line
        # displacement from the current sample to the target.
        return z - x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional score for p_t(x|z).
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - s_t: conditional score (bs, p)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)

        return (z * alpha_t - x) / beta_t**2


class SO2GaussianCPP(ConditionalProbabilityPath):
    """
    Gaussian conditional probability path but constrained to(-pi, pi).
    """

    def __init__(self, cfg, alpha: Scheduler, beta: Scheduler):
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta

    def sample_x_init(self, z: torch.Tensor) -> torch.Tensor:
        """
        Samples the initial point x from the distribution p_0(x|z).
        Args:
            - z: gt value (bs, p)
        Returns:
            - x0: sampled initial point (bs, p)
        """
        if self.cfg.dataset.normalize_q:
            return torch.rand_like(z) * 2.0 - 1.0
        return torch.fmod(torch.randn_like(z) * 0.05, math.pi)

    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, x0: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns samples from the Gaussian distribution p_t(x|z).
        Args:
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - x: samples from dist (bs, p)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)

        # Sample noise
        if x0 is None:
            noise = self.sample_x_init(z)
        else:
            noise = x0

        # Compute conditional path
        z_conditional = alpha_t * z + beta_t * noise
        z_conditional = torch.atan2(torch.sin(z_conditional), torch.cos(z_conditional))
        return z_conditional, noise

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z) for a Gaussian probability path.
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - u_t: conditional vector field (bs, p)
        """
        # Linear schedule: the conditional vector field is the straight-line
        # displacement from the current sample to the target (see R3GaussianCPP).
        return z - x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional score for p_t(x|z).
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - s_t: conditional score (bs, p)
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)

        s_t = (z * alpha_t - x) / beta_t**2
        return s_t


class SO3GaussianCPP(ConditionalProbabilityPath):
    """
    Class based on https://github.com/DreamFold/FoldFlow/blob/main/FoldFlow/so3/so3_flow_matching.py
    """

    def sample_x_init(self, z: torch.Tensor) -> torch.Tensor:
        """
        Samples the initial point x from the distribution p_0(x|z).
        Args:
            - z: gt value (bs, any)
        Returns:
            - x0: sampled initial point (bs, 3, 3)
        """
        # For SO(3), we can sample uniformly on the manifold
        rotation = R.random(z.shape[0])
        x_init = rotation.as_matrix()
        x_init = torch.tensor(x_init, dtype=torch.float32, device=z.device)
        return x_init

    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, x0: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Returns samples from the distribution p_t(x|z).
        Args:
            - z: gt value (bs, 6)
            - t: time (bs, 1)
        Returns:
            - x: samples from dist (bs, 6)
        """
        z_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(z)
        if x0 is None:
            x_init_9d = self.sample_x_init(z_9d)  # (bs, 3, 3)
        else:
            x_init_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(x0)

        z_quat = math_utils.quat_from_matrix(z_9d)
        x_init_quat = math_utils.quat_from_matrix(x_init_9d)

        z_SO3 = pp.SO3(z_quat)  # (x, y, z, w)
        x_init_SO3 = pp.SO3(x_init_quat)

        log_zx = (x_init_SO3.Inv() @ z_SO3).Log()  # (bs, 3)
        t_log = t * log_zx  # (bs, 3)
        xt = x_init_SO3 @ pp.so3(t_log).Exp()  # (bs, 4)
        xt = math_utils.matrix_from_quat(xt)  # (bs, 3, 3)
        xt = xt[:, :, :2].transpose(1, 2).reshape(xt.shape[0], -1)  # (bs, 6)
        x_init_6d = x_init_9d[:, :, :2].transpose(1, 2).reshape(x_init_9d.shape[0], -1)  # (bs, 6)
        return xt, x_init_6d

    def conditional_vector_field(self, x, z, t):
        """
        Returns the conditional vector field u_t(x|z) for a Gaussian probability path.
        Args:
            - x: input value (bs, 6)
            - z: gt value (bs, 6)
            - t: time (bs, 1)
        Returns:
            - u_t: conditional vector field (bs, 6)
        """
        x_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(x)
        z_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(z)
        x_quat = math_utils.quat_from_matrix(x_9d)
        z_quat = math_utils.quat_from_matrix(z_9d)
        x_SO3 = pp.SO3(x_quat)  # (x, y, z, w)
        z_SO3 = pp.SO3(z_quat)

        ut_ref = (x_SO3.Inv() @ z_SO3).Log()
        return ut_ref

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass


class SE3GaussianCPP(ConditionalProbabilityPath):
    """
    Gaussian conditional probability path.
    """

    def __init__(self, cfg, alpha: Scheduler, beta: Scheduler):
        self.cfg = cfg
        self.alpha = alpha
        self.beta = beta

        self.r3_conditional_path = R3GaussianCPP(cfg, alpha, beta)  # For translation
        self.so3_conditional_path = SO3GaussianCPP()  # For orientation R3GaussianCPP(alpha, beta) #
        self.so2_conditional_path = SO2GaussianCPP(cfg, alpha, beta)  # For joint angles

    def sample_x_init(self, z: torch.Tensor) -> torch.Tensor:
        """
        Samples the initial point x from the distribution p_0(x|z).
        Args:
            - z: gt value (bs, p)
        Returns:
            - x0: sampled initial point (bs, p)
        """
        # Split z into translation, orientation, and joint angles
        z_translation, z_orientation, z_joint_angles = self.split_sample(z)

        # Sample translation and orientation
        x_translation = self.r3_conditional_path.sample_x_init(z_translation)
        x_orientation = (
            self.so3_conditional_path.sample_x_init(z_orientation)[:, :, :2]
            .transpose(1, 2)
            .reshape(z.shape[0], -1)
        )  # Convert to 6D rotation form
        x_joint_angles = self.so2_conditional_path.sample_x_init(z_joint_angles)

        # Combine samples
        x = torch.cat([x_translation, x_orientation, x_joint_angles], dim=-1)
        return x

    def split_sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits z into translation, orientation, and joint angles.
        Args:
            - z: input tensor (bs, p)
        Returns:
            - z_translation: translation part (bs, 3)
            - z_orientation: orientation part (bs, 6)
            - z_joint_angles: joint angles part (bs, p - 9)
        """
        z_translation = z[:, :3]  # Assuming first 3 dimensions are translation
        z_orientation = z[:, 3:9]  # Assuming next 6 dimensions are orientation
        z_joint_angles = z[:, 9:]  # Assuming remaining dimensions are joint angles
        return z_translation, z_orientation, z_joint_angles

    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, x0: Dict = None
    ) -> torch.Tensor:
        """
        Returns samples from the Gaussian distribution p_t(x|z).
        Args:
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - x: samples from dist (bs, p)
        """
        z_translation, z_orientation, z_joint_angles = self.split_sample(z)

        # Sample translation and orientation
        if x0 is None:
            x0_r3, x0_so3, x0_so2 = None, None, None
        else:
            x0_r3, x0_so3, x0_so2 = self.split_sample(x0)
        x_translation, x0_r3 = self.r3_conditional_path.sample_conditional_path(
            z_translation, t, x0=x0_r3
        )
        x_orientation, x0_so3 = self.so3_conditional_path.sample_conditional_path(
            z_orientation, t, x0=x0_so3
        )
        x_joint_angles, x0_so2 = self.so2_conditional_path.sample_conditional_path(
            z_joint_angles, t, x0=x0_so2
        )

        # Combine samples
        x = torch.cat([x_translation, x_orientation, x_joint_angles], dim=-1)
        x0 = torch.cat([x0_r3, x0_so3, x0_so2], dim=-1)
        return x, x0

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z) for a Gaussian probability path.
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - u_t: conditional vector field (bs, p)
        """
        z_translation, z_orientation, z_joint_angles = self.split_sample(z)
        x_translation, x_orientation, x_joint_angles = self.split_sample(x)

        u_translation = self.r3_conditional_path.conditional_vector_field(
            x_translation, z_translation, t
        )
        u_orientation = self.so3_conditional_path.conditional_vector_field(
            x_orientation, z_orientation, t
        )
        u_joint_angles = self.so2_conditional_path.conditional_vector_field(
            x_joint_angles, z_joint_angles, t
        )

        u_t = torch.cat([u_translation, u_orientation, u_joint_angles], dim=-1)
        return u_t

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional score for p_t(x|z).
        Args:
            - x: input value (bs, p)
            - z: gt value (bs, p)
            - t: time (bs, 1)
        Returns:
            - s_t: conditional score (bs, p)
        """
        z_translation, z_orientation, z_joint_angles = self.split_sample(z)
        x_translation, x_orientation, x_joint_angles = self.split_sample(x)

        s_translation = self.r3_conditional_path.conditional_score(x_translation, z_translation, t)
        s_orientation = self.so3_conditional_path.conditional_score(x_orientation, z_orientation, t)
        s_joint_angles = self.so2_conditional_path.conditional_score(
            x_joint_angles, z_joint_angles, t
        )

        s_t = torch.cat([s_translation, s_orientation, s_joint_angles], dim=-1)
        return s_t


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        assert out_dim % 2 == 0
        self.half_dim = out_dim // 2
        self.weights = nn.Parameter(torch.randn(in_dim, self.half_dim, dtype=torch.float32) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes x into a Fourier feature representation.
        Args:
            - x  (bs, samples, in_dim)
        Returns:
            - encoded_x: Fourier encoded time (bs, samples, out_dim)
        """
        freqs = x @ self.weights * 2 * math.pi  # (bs, samples, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, samples, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, samples, half_dim)
        encoded_x = torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(
            2
        )  # (bs, samples, out_dim)
        return encoded_x


######################### Flow Matching Model #########################


class TransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # Self-attention
        self.sattn_norm = nn.LayerNorm(512)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=self.cfg.model.num_heads_xattn, batch_first=True
        )

        # Cross-attention
        self.xattn_norm_robot = nn.LayerNorm(512)
        self.cross_attn_robot_cond = nn.MultiheadAttention(
            embed_dim=512, num_heads=self.cfg.model.num_heads_xattn, batch_first=True
        )

        self.xattn_norm_scene = nn.LayerNorm(512)
        self.cross_attn_scene_cond = nn.MultiheadAttention(
            embed_dim=512, num_heads=self.cfg.model.num_heads_xattn, batch_first=True
        )

        # FFN
        self.ffn_norm = nn.LayerNorm(512)
        self.ffn = MLP(512, 512, 2, 512, use_layer_norm=True)

    def forward(self, action_tokens, robot_tokens, scene_tokens):
        # self-attn (LayerNorm works on last dim, so apply to full tensor)
        action_tokens_norm = self.sattn_norm(action_tokens)
        sattn_out, _ = self.self_attn(action_tokens_norm, action_tokens_norm, action_tokens_norm)
        action_tokens = action_tokens + sattn_out

        # Cross-attention with robot tokens
        action_tokens_norm = self.xattn_norm_robot(action_tokens)
        x_attn_out, _ = self.cross_attn_robot_cond(action_tokens_norm, robot_tokens, robot_tokens)
        action_tokens = action_tokens + x_attn_out

        # Cross-attention with scene tokens
        action_tokens_norm = self.xattn_norm_scene(action_tokens)
        x_attn_out, _ = self.cross_attn_scene_cond(action_tokens_norm, scene_tokens, scene_tokens)
        action_tokens = action_tokens + x_attn_out

        # FFN
        action_tokens_norm = self.ffn_norm(action_tokens)
        ffn_out = self.ffn(action_tokens_norm)
        action_tokens = action_tokens + ffn_out

        return action_tokens


class SeededGraspFlow(nn.Module):
    def __init__(self, cfg, robot_models=None, robot_models_path=None) -> None:
        super().__init__()
        self.cfg = cfg
        self._warned_unknown_alignment_grippers = set()
        self._use_rotation_alignment = bool(
            getattr(self.cfg.dataset, "gripper_alignment_rot6d", None)
        )
        self._alignment_transforms = build_gripper_alignment_transforms(
            gripper_alignment_rot6d=getattr(self.cfg.dataset, "gripper_alignment_rot6d", None),
            canonical_rot6d=getattr(self.cfg.dataset, "canonical_rot6d", None),
        )

        # Initialize flow matching structure
        self.alpha_t = LinearAlpha()  # hydrize
        self.beta_t = LinearBeta()  # hydrize
        self.cond_prob_path = SE3GaussianCPP(  # hydrize
            cfg=cfg, alpha=self.alpha_t, beta=self.beta_t
        )

        # Input dimensions
        self.point_cloud_embeddings_dim = self.cfg.model.point_cloud_embeddings_dim
        self.time_dim = self.cfg.model.time_dim
        self.query_dim = self.cfg.model.query_dim

        # Initialize time embedder
        self.time_embedder = FourierEncoder(1, 128)  # (bs, time_dim)
        self.time_mlp = MLP(128, self.time_dim, 2, 512, use_layer_norm=True)

        # Initialize learnable query
        self.query_dict = nn.ParameterDict()
        for i in self.cfg.dataset.robot_name_list:
            name = cfg.dataset.robot_name_mapping[i]
            self.query_dict[name] = nn.Parameter(torch.randn(1, self.query_dim))
        self.query_dict = self.query_dict.to(device)

        # Create robot name to index mapping for DataParallel compatibility
        # (lists don't get split properly by DataParallel, but tensors do)
        self.robot_name_to_idx = {}
        self.idx_to_robot_name = {}
        for idx, i in enumerate(self.cfg.dataset.robot_name_list):
            name = cfg.dataset.robot_name_mapping[i]
            self.robot_name_to_idx[name] = idx
            self.idx_to_robot_name[idx] = name

        # Initialize robot models
        self.robot_models = {} if robot_models is None else robot_models
        self.robot_adj = {}
        for i in self.cfg.dataset.robot_name_list:
            robot_name = cfg.dataset.robot_name_mapping[i]
            if robot_name not in self.robot_models:
                self.robot_models[robot_name] = get_handmodel(
                    robot_name,
                    1,
                    device,
                    1.0,
                    data_dir=to_absolute_path(cfg.dataset.dataset_basedir),
                )
            # robot_models are already on the correct device (loaded with map_location=device)
            joint_lower = np.array(
                self.robot_models[robot_name].revolute_joints_q_lower.cpu().reshape(-1)
            )
            joint_upper = np.array(
                self.robot_models[robot_name].revolute_joints_q_upper.cpu().reshape(-1)
            )
            joint_mid = (joint_lower + joint_upper) / 2
            joints_q = (joint_mid + joint_lower) / 2
            rest_pose = (
                torch.from_numpy(np.concatenate([np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), joints_q]))
                .unsqueeze(0)
                .to(device)
                .float()
            )
            surface_points = (
                self.robot_models[robot_name]
                .get_surface_points(q=rest_pose, downsample=True, num_points=cfg.model.robot_pc_n)
                .cpu()
                .squeeze(0)
            )
            self.robot_adj[robot_name], _ = generate_adj_mat_feats(
                surface_points, knn=cfg.dataset.knn
            )
            self.robot_adj[robot_name] = self.robot_adj[robot_name].to(device)
        if robot_models is None:
            torch.save(self.robot_models, "robot_features.pt")

        # Device-specific caches for DataParallel compatibility
        # Load separate copies of robot models for each GPU from the saved file
        self._robot_models_cache = {}
        self._robot_adj_cache = {}
        self._robot_models_path = robot_models_path

        # Pre-initialize caches for all available CUDA devices
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            gpu_device = torch.device(f"cuda:{gpu_id}")
            # Load a fresh copy of hand models for this specific GPU
            if robot_models_path is not None:
                gpu_robot_models = torch.load(
                    robot_models_path, weights_only=False, map_location=gpu_device
                )

            for rname in self.robot_models:
                cache_key = (rname, str(gpu_device))
                self._robot_models_cache[cache_key] = gpu_robot_models[rname]
                # Pre-cache adjacency matrices for each device
                self._robot_adj_cache[cache_key] = self.robot_adj[rname].to(gpu_device)

        # Padded joint-angle width: pad_q covers 9 pose dims (translation +
        # rot6d) plus the largest joint count across the configured grippers.
        self.num_dof_slots = self.cfg.dataset.pad_q - 9

        # Initialize x embedder
        self.xt_embedder = MLP(3, self.cfg.model.x_embed_dim, 2, 512, use_layer_norm=True)
        self.xr_embedder = MLP(3, self.cfg.model.x_embed_dim, 2, 512, use_layer_norm=True)
        self.xd_embedder = MLP(
            self.num_dof_slots, self.cfg.model.x_embed_dim, 2, 512, use_layer_norm=True
        )

        # Initialize point cloud encoder
        self.obj_in_feats = self.cfg.model.obj_in_feats
        self.robot_in_feats = self.cfg.model.robot_in_feats
        self.hidden_n = self.cfg.model.hidden_n
        self.obj_out_feats = self.point_cloud_embeddings_dim
        self.robot_out_feats = self.point_cloud_embeddings_dim
        self.gcn_dropout = self.cfg.model.gcn_dropout
        self.gcn_num_hidden = self.cfg.model.gcn_num_hidden

        self.obj_pc_pos_encoder = FourierEncoder(3, self.obj_in_feats)
        self.robot_pc_pos_encoder = FourierEncoder(3, self.robot_in_feats)

        self.relative_seed_encoder = FourierEncoder(3, 32)

        self.obj_normals_encoder = FourierEncoder(3, 32)
        self.obj_curvature_encoder = FourierEncoder(1, 32)

        self.obj_encoder = GCN(  # (bs, obj_pc_n, obj_out_feats)
            nfeat=self.obj_in_feats + 96,
            nhid=self.hidden_n,
            nout=self.obj_out_feats,
            dropout=self.gcn_dropout,
            num_hidden=self.gcn_num_hidden,
        )
        self.robot_encoder = GCN(  # (bs, robot_pc_n, robot_out_feats)
            nfeat=self.robot_in_feats,
            nhid=self.hidden_n,
            nout=self.robot_out_feats,
            dropout=self.gcn_dropout,
            num_hidden=self.gcn_num_hidden,
        )

        self.local_mlp = MLP(self.point_cloud_embeddings_dim + 3, 512, 3, 512, use_layer_norm=True)

        self.trans_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(2)])

        self.net_t = MLP(512, 3, 3, 512, use_layer_norm=False)
        self.net_r = MLP(512, 3, 3, 1024, use_layer_norm=False)
        self.net_d = MLP(512, self.num_dof_slots, 3, 512, use_layer_norm=False)

    def _get_robot_model_for_device(self, rname: str, device: torch.device):
        """Get a device-specific copy of the robot model.

        Returns pre-initialized robot model for the specific device.
        Models are created during __init__ to avoid race conditions in DataParallel.
        """
        cache_key = (rname, str(device))
        if cache_key not in self._robot_models_cache:
            # Fallback: load on-demand if not pre-cached (shouldn't happen normally)
            if self._robot_models_path is not None:
                gpu_robot_models = torch.load(
                    self._robot_models_path, weights_only=False, map_location=device
                )
                self._robot_models_cache[cache_key] = gpu_robot_models[rname]
            else:
                from hydra.utils import to_absolute_path

                self._robot_models_cache[cache_key] = get_handmodel(
                    rname,
                    1,
                    device,
                    1.0,
                    data_dir=to_absolute_path(self.cfg.dataset.dataset_basedir),
                )
        return self._robot_models_cache[cache_key]

    def _get_robot_adj_for_device(self, rname: str, device: torch.device):
        """Get a device-specific copy of the robot adjacency matrix.

        Caches copies per device to avoid repeated .to() calls and ensure
        thread-safety when using DataParallel.
        """
        cache_key = (rname, str(device))
        if cache_key not in self._robot_adj_cache:
            self._robot_adj_cache[cache_key] = self.robot_adj[rname].to(device)
        return self._robot_adj_cache[cache_key]

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obj_pc: torch.Tensor,
        obj_adj: torch.Tensor,
        robot_name_idx: torch.Tensor,
        seed_pt: torch.tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Decode robot name indices to strings (for DataParallel compatibility)
        # robot_name_idx is a tensor that gets properly split by DataParallel
        robot_name = [self.idx_to_robot_name[idx.item()] for idx in robot_name_idx]

        # Apply classifier-free guidance dropout: zero out object and seed point conditioning for dropped samples
        if drop_mask is not None and drop_mask.any():
            obj_pc = obj_pc.clone()
            obj_adj = obj_adj.clone()
            seed_pt = seed_pt.clone()
            obj_pc[drop_mask] = 0.0
            obj_adj[drop_mask] = 0.0
            seed_pt[drop_mask] = 0.0

        # Find closest point
        dists = torch.cdist(seed_pt, obj_pc)
        knn_idx = dists.topk(k=16, largest=False).indices.squeeze(1)
        neighbour_pts = torch.gather(obj_pc, 1, knn_idx.unsqueeze(-1).expand(-1, -1, 3))
        relative_xyz = neighbour_pts - x[:, 0:3].unsqueeze(1)

        relative_seed = obj_pc - neighbour_pts[:, 0, :].unsqueeze(1)
        relative_seed = self.relative_seed_encoder(relative_seed)

        # Add in robot? (Optimized batched version)
        batch_size = x.shape[0]

        # Group samples by robot type for batched FK computation
        robot_groups = {}
        for i, rname in enumerate(robot_name):
            if rname not in robot_groups:
                robot_groups[rname] = []
            robot_groups[rname].append(i)

        # Preallocate tensors (get shape from first robot)
        robot_pc_list = [None] * batch_size

        # Process each robot type in batches
        for rname, indices in robot_groups.items():
            # Use tensor indexing instead of list comprehension
            indices_tensor = torch.tensor(indices, device=x.device)
            dof_len = len(self.cfg.dataset.dof_mapping[rname])
            q_batch = x[indices_tensor, : 9 + dof_len]
            # Unnormalize q_batch before FK using helper
            q_batch_unnorm = unnormalize_q(self.cfg, q_batch, [rname] * q_batch.shape[0])
            if self._use_rotation_alignment:
                q_batch_unnorm = apply_gripper_rotation_alignment_with_transforms(
                    q=q_batch_unnorm,
                    gripper_id=rname,
                    transforms=self._alignment_transforms,
                    inverse=True,
                    warned_unknown_grippers=self._warned_unknown_alignment_grippers,
                )
            # Get device-cached robot model for DataParallel compatibility
            robot_model = self._get_robot_model_for_device(rname, x.device)
            # Single batched FK call for all samples with this robot
            pc_batch = robot_model.get_surface_points(
                q=q_batch_unnorm, downsample=True, num_points=self.cfg.model.robot_pc_n
            )
            # Normalize robot_pc to [-1, 1] like obj_pc using helper
            pc_batch = normalize_pc(self.cfg, pc_batch)
            # Assign results back to original positions
            for idx_in_batch, orig_idx in enumerate(indices):
                robot_pc_list[orig_idx] = pc_batch[idx_in_batch]

        # Build output tensors - use list comprehensions (faster than append loops)
        query_tensor = torch.stack([self.query_dict[rname].to(x.device) for rname in robot_name])
        robot_pc = torch.stack(robot_pc_list)
        robot_adj_list = [self._get_robot_adj_for_device(rname, x.device) for rname in robot_name]
        # Convert to dense before stacking so we can mask and use in GNN layers
        robot_adj = torch.stack([a.to_dense() if a.is_sparse else a for a in robot_adj_list])

        # Apply classifier-free guidance dropout: zero out robot conditioning for dropped samples
        if drop_mask is not None and drop_mask.any():
            robot_pc = robot_pc.clone()
            robot_adj = robot_adj.clone()
            query_tensor = query_tensor.clone()
            robot_pc[drop_mask] = 0.0
            robot_adj[drop_mask] = 0.0
            query_tensor[drop_mask] = 0.0

        # Convert 6D to axis angle
        x_9d = math_utils.robust_compute_rotation_matrix_from_ortho6d(x[:, 3:9])
        x_quat = math_utils.quat_from_matrix(x_9d)
        x_SO3 = pp.SO3(x_quat)
        x_aa = x_SO3.Log()
        x = torch.cat((x[:, 0:3], x_aa, x[:, 9:]), dim=1)

        # Sample t and embed
        t_embed = self.time_embedder(t)  # (bs, time_dim)
        t_embed = self.time_mlp(t_embed)

        # x embedder
        x_t = x[:, 0:3]
        x_r = x[:, 3:6]
        x_d = x[:, 6:]
        x_t = self.xt_embedder(x_t)
        x_r = self.xr_embedder(x_r)
        x_d = self.xd_embedder(x_d)

        # Add in time - stack is faster than cat with unsqueezes
        action_tokens = torch.stack([x_t, x_r, x_d], dim=1)  # (bs, 3, dim)
        action_tokens = action_tokens + t_embed.unsqueeze(1)

        # Additional point cloud processing
        normals, curvature = compute_gradient_and_curvature(obj_pc, obj_adj)
        normals = self.obj_normals_encoder(normals)
        curvature = self.obj_curvature_encoder(curvature.unsqueeze(-1))

        # Point cloud encoder
        obj_pc = self.obj_pc_pos_encoder(obj_pc)  # (bs, obj_pc_n, obj_in_feats)
        robot_pc = self.robot_pc_pos_encoder(robot_pc)  # (bs, robot_pc_n, robot_in_feats)

        obj_pc = torch.cat([obj_pc, relative_seed, normals, curvature], dim=2)
        obj_embed = self.obj_encoder(obj_pc, obj_adj)  # (bs, obj_pc_n, obj_out_feats)
        robot_embed = self.robot_encoder(robot_pc, robot_adj)  # (bs, robot_pc_n, robot_out_feats)

        # Apply max pooling to get global graph features
        obj_feat_max = torch.amax(obj_embed, dim=1)  # (bs, point_cloud_embeddings_dim)
        robot_feat_max = torch.amax(robot_embed, dim=1)  # (bs, point_cloud_embeddings_dim)
        obj_feat_mean = torch.mean(obj_embed, dim=1)
        robot_feat_mean = torch.mean(robot_embed, dim=1)

        # Get local graph features
        neighbour_feats = torch.gather(obj_embed, 1, knn_idx.unsqueeze(-1).expand(-1, -1, 512))
        local_input = torch.cat([relative_xyz, neighbour_feats], dim=-1)  # (bs, k, 3+D)
        local_embedding = self.local_mlp(local_input)

        robot_tokens = torch.cat(
            [robot_feat_max.unsqueeze(1), robot_feat_mean.unsqueeze(1), query_tensor], dim=1
        )
        scene_tokens = torch.cat(
            [obj_feat_max.unsqueeze(1), obj_feat_mean.unsqueeze(1), local_embedding], dim=1
        )
        for block in self.trans_blocks:
            action_tokens = block(action_tokens, robot_tokens, scene_tokens)

        u_t = self.net_t(action_tokens[:, 0, :])
        u_r = self.net_r(action_tokens[:, 1, :])
        u_d = self.net_d(action_tokens[:, 2, :])

        return torch.cat([u_t, u_r, u_d], dim=1)

    def calc_loss(
        self,
        ut_theta: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        e: int,
    ) -> torch.Tensor:
        mask = ~torch.isnan(z[:, 9:])
        ut_ref = self.cond_prob_path.conditional_vector_field(x0, z, t)  # (bs, p)

        trans_err, rot_err, joint_err = [], [], []
        for i in range(z.shape[0]):
            valid_idx = mask[i]
            trans_err.append((ut_theta[i][:3] - ut_ref[i][:3]).abs().sum())
            rot_err.append((ut_theta[i][3:6] - ut_ref[i][3:6]).abs().sum())
            joint_err.append((ut_theta[i][6:][valid_idx] - ut_ref[i][6:][valid_idx]).abs().mean())

        trans_err = torch.stack(trans_err)
        rot_err = torch.stack(rot_err)
        joint_err = torch.stack(joint_err)

        # L = lambda_trans * L_trans + lambda_rot * L_rot + lambda_joints * L_joints
        tot_err = (
            self.cfg.train.lambda_trans * trans_err
            + self.cfg.train.lambda_rot * rot_err
            + self.cfg.train.lambda_joints * joint_err
        )
        return tot_err.mean(), (trans_err.mean(), rot_err.mean(), joint_err.mean())
