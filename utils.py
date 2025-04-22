import torch

def transform_points(points, trans, device):
    # Transform points in SE(3)        points:(B,N,3)       trans:(B,N,4,4)
    B,N = points.shape[0], points.shape[1]
    ones = torch.ones([B,N, 1], device=device).float()
    points_ = torch.cat([points, ones], dim=-1)  # Convert to homogeneous coordinates
    # Perform the matrix multiplication
    trans_points = torch.einsum('bnij,bnj->bni', trans, points_)
    # Return the transformed points (excluding the homogeneous coordinate)
    return trans_points[:, :,:3]

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    # w,x,y,z
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:

    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# Logarithmic map for R^3 x S^3 manifold (with e in tangent space)
def logmap_th(f, f0):
    H = dQuatToDxJac_th(f0)  # Shape: (N, 3, 4)
    log_s3 = logmap_S3_th(f, f0)  # Shape: (N, 4)
    e = 2 * torch.einsum('ijk,ik->ij', H, log_s3)  # Batch matrix multiplication with einsum
    return e

# Logarithmic map for S^3 manifold (with e in ambient space)
def logmap_S3_th(x, x0):
    dot_product = torch.einsum('ij,ij->i', x0, x)  # Shape: (N,)
    th = acoslog_th(dot_product)
    u = x - dot_product.unsqueeze(-1)*x0
    u = (th.unsqueeze(-1)*u) / (torch.norm(u,dim=-1) + 1e-4).unsqueeze(-1)
    return u

# Arcosine redefinition to ensure distance between antipodal quaternions is zero
def acoslog_th(x):
    y = torch.acos(torch.clamp(x,-1+1e-2,1-1e-2))
    mask = (x >= -1.0) & (x < 0)
    y[mask] = y[mask] - torch.pi
    return y

def dQuatToDxJac_th(q):
    # Create the Jacobian matrix for each quaternion in the batch
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]  # Shape: (N,)
    H = torch.stack([
        -q1, q0, -q3, q2,
        -q2, q3, q0, -q1,
        -q3, -q2, q1, q0
    ], dim=1).reshape(-1, 3, 4)  # Shape: (N, 3, 4)
    return H

def quat_multiply(q1, q2, q_res):
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]

    q_res[..., 0] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z

    q_res[..., 1] = a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z
    q_res[..., 2] = a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x
    q_res[..., 3] = a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y
    return q_res

def geodesic_distance(goal_quat, current_quat, quat_res):
    conjugate_quat = current_quat.detach().clone()
    conjugate_quat[..., 1:] *= -1.0
    quat_res = quat_multiply(goal_quat, conjugate_quat, quat_res)
    sign = torch.sign(quat_res[..., 0])
    sign = torch.where(sign == 0, 1.0, sign)
    quat_res = -1.0 * quat_res * sign.unsqueeze(-1)
    quat_res[..., 0] = 0.0
    rot_error = torch.norm(quat_res, dim=-1, keepdim=True)
    scale = 1.0 / rot_error
    scale = torch.nan_to_num(scale, 0.0, 0.0, 0.0)
    quat_res = quat_res * scale
    return quat_res, rot_error

def geodesic_distance_between_quaternions(
    q1: torch.Tensor, q2: torch.Tensor, acos_epsilon= None
) -> torch.Tensor:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """
    # Note: Decreasing this value to 1e-8 greates NaN gradients for nearby quaternions.
    acos_clamp_epsilon = 1e-7
    if acos_epsilon is not None:
        acos_clamp_epsilon = acos_epsilon

    dot = torch.clip(torch.sum(q1 * q2, dim=1), -1, 1)
    distance = 2 * torch.acos(torch.clamp(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon))
    distance = torch.abs(torch.remainder(distance + torch.pi, 2 * torch.pi) - torch.pi)  # TODO: do we need this?
    return distance

def SO3_Riemannian_metric(q1):
    M = torch.eye(4, device=q1.device).unsqueeze(0).repeat(q1.shape[0], 1, 1) - torch.einsum('bi,bj->bij', q1, q1)
    return M

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles