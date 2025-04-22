import torch
import os
import utils
from robot import RobotModel
import time
import trimesh

def GaussNewton(rbt,current_q,goal,line_search=0.2,iters=50):
    for i in range(iters):
        pos,quat = rbt.forward_kinematics_eef(current_q)
        J_pos,J_quat = rbt.get_Jacobian(current_q)
        J_quat = J_quat*0.1
        J = torch.cat([J_pos,J_quat],dim=1)
        error = torch.cat([pos-goal[:,:3],utils.logmap_th(quat,goal[:,3:])],dim=-1)
        J_pinv = torch.linalg.pinv(J)
        grad = torch.einsum('ijk,ik->ij', J_pinv,error)
        current_q = current_q - line_search * grad
    return current_q

def run_ik(rbt, goal, n_samples=1000, pos_error_threshold=0.03, ori_error_threshold=0.3):
    t0 = time.time()
    # random initial guess
    q = torch.rand([n_samples, rbt.n_joints],requires_grad=True).to(device) * (rbt.theta_max -rbt.theta_min) + rbt.theta_min
    # solve IK
    result = GaussNewton(rbt, q, goal)
    # check if the solution is valid
    pos, quat = rbt.forward_kinematics_eef(result)
    pos_mask = torch.norm(pos-goal[:,:3], dim=1) < pos_error_threshold
    ori_mask = utils.geodesic_distance_between_quaternions(quat, goal[:,3:]) < ori_error_threshold
    mask = pos_mask & ori_mask
    result = result[mask]
    # limit joint angles from -pi to pi
    result = rbt.limit_angles(result)
    # check joint limits
    joint_limits_mask = (result>rbt.theta_min.unsqueeze(0)).all(dim=1) & (result<rbt.theta_max.unsqueeze(0)).all(dim=1)
    result = result[joint_limits_mask]
    t1 = time.time()
    print(f"Valid solutions: {result.shape[0]}, time cost: {t1-t0}")
    return result

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize = True

    if visualize:
        load_mesh = True
    else:
        load_mesh = False
    rbt = RobotModel(
        urdf_path=os.path.join('urdf/panda_urdf/panda.urdf'),
        last_link_name="panda_hand",
        load_mesh=load_mesh,
        device=device
    )

    # x: (N,7) (pos,quat)
    n_samples = 1000
    x = torch.tensor([[0.5,0.0,0.3,0.,0.,0.,1.0]]).to(device).expand(n_samples,-1)
    q = run_ik(rbt, x, n_samples=n_samples, pos_error_threshold=0.03, ori_error_threshold=0.3)

    if visualize:
        scene = trimesh.Scene()
        random_idx = torch.randint(0, len(q), (10,))
        q4vis = q[random_idx]

        for _q in q4vis:
            mesh = rbt.theta2mesh(_q.clone().unsqueeze(0))
            scene.add_geometry(mesh)
        scene.show()