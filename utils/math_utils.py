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

"""Math utilities for rotations and vector algebra."""

import numpy as np
import torch
import transforms3d


def get_rot6d_from_quat(quat):
    rotation_matrix = transforms3d.quaternions.quat2mat(quat)
    return rotation_matrix.T.reshape(-1)[:6]


def get_rot6d_from_rot3d(rot3d):
    global_rotation = np.array(transforms3d.euler.euler2mat(rot3d[0], rot3d[1], rot3d[2]))
    return global_rotation.T.reshape(9)[:6]


def quat_from_matrix(matrix):
    """Convert a rotation matrix to a quaternion."""
    m = matrix
    qw = torch.empty(matrix.shape[:-2], device=matrix.device)
    qx = torch.empty_like(qw)
    qy = torch.empty_like(qw)
    qz = torch.empty_like(qw)

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    # Case 1: trace is positive
    positive_trace = trace > 0
    t = torch.sqrt(trace[positive_trace] + 1.0)
    qw[positive_trace] = 0.5 * t
    qx[positive_trace] = (m[..., 2, 1][positive_trace] - m[..., 1, 2][positive_trace]) / (2.0 * t)
    qy[positive_trace] = (m[..., 0, 2][positive_trace] - m[..., 2, 0][positive_trace]) / (2.0 * t)
    qz[positive_trace] = (m[..., 1, 0][positive_trace] - m[..., 0, 1][positive_trace]) / (2.0 * t)

    # Case 2: diagonal term m[0][0] is largest
    cond0 = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]) & ~positive_trace
    t = torch.sqrt(1.0 + m[..., 0, 0][cond0] - m[..., 1, 1][cond0] - m[..., 2, 2][cond0])
    qx[cond0] = 0.5 * t
    qw[cond0] = (m[..., 2, 1][cond0] - m[..., 1, 2][cond0]) / (2.0 * t)
    qy[cond0] = (m[..., 0, 1][cond0] + m[..., 1, 0][cond0]) / (2.0 * t)
    qz[cond0] = (m[..., 0, 2][cond0] + m[..., 2, 0][cond0]) / (2.0 * t)

    # Case 3: diagonal term m[1][1] is largest
    cond1 = (m[..., 1, 1] > m[..., 2, 2]) & ~positive_trace & ~cond0
    t = torch.sqrt(1.0 + m[..., 1, 1][cond1] - m[..., 0, 0][cond1] - m[..., 2, 2][cond1])
    qy[cond1] = 0.5 * t
    qw[cond1] = (m[..., 0, 2][cond1] - m[..., 2, 0][cond1]) / (2.0 * t)
    qx[cond1] = (m[..., 0, 1][cond1] + m[..., 1, 0][cond1]) / (2.0 * t)
    qz[cond1] = (m[..., 1, 2][cond1] + m[..., 2, 1][cond1]) / (2.0 * t)

    # Case 4: diagonal term m[2][2] is largest
    cond2 = ~positive_trace & ~cond0 & ~cond1
    t = torch.sqrt(1.0 + m[..., 2, 2][cond2] - m[..., 0, 0][cond2] - m[..., 1, 1][cond2])
    qz[cond2] = 0.5 * t
    qw[cond2] = (m[..., 1, 0][cond2] - m[..., 0, 1][cond2]) / (2.0 * t)
    qx[cond2] = (m[..., 0, 2][cond2] + m[..., 2, 0][cond2]) / (2.0 * t)
    qy[cond2] = (m[..., 1, 2][cond2] + m[..., 2, 1][cond2]) / (2.0 * t)

    quat = torch.stack((qx, qy, qz, qw), dim=-1)

    # Enforce w >= 0 convention to handle quaternion double cover
    w_negative = quat[..., -1] < 0
    quat = torch.where(w_negative.unsqueeze(-1), -quat, quat)

    return quat


def matrix_from_quat(q):
    # Normalize the quaternions to ensure unit norm
    q = torch.nn.functional.normalize(q, p=2, dim=-1)

    # Enforce w >= 0 convention to handle quaternion double cover
    w_negative = q[..., -1] < 0
    q = torch.where(w_negative.unsqueeze(-1), -q, q)

    x, y, z, w = q.unbind(-1)

    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_matrix = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )  # shape (..., 3, 3)

    return rot_matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """Recover a rotation matrix from a continuous 6D rotation representation.

    Follows the 6D parameterisation of Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks" (CVPR 2019), via the GeoMatch
    implementation, with an added Gram-Schmidt fallback for near-parallel
    input vectors.
    """

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw)

    # Check for near-parallel vectors
    dot_product = (x * y).sum(dim=1).abs()
    parallel_threshold = 0.99

    if (dot_product > parallel_threshold).any():
        # For near-parallel cases, use a more stable Gram-Schmidt
        y = y - (x * y).sum(dim=1, keepdim=True) * x
        y = normalize_vector(y)
        z = normalize_vector(cross_product(x, y))
    else:
        # Original robust method for non-parallel cases
        middle = normalize_vector(x + y)
        orthmid = normalize_vector(x - y)
        x = normalize_vector(middle + orthmid)
        y = normalize_vector(middle - orthmid)
        z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out
