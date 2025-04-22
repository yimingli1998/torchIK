import torch
import pytorch_kinematics as pk
import utils
import glob
import os
import trimesh
import copy
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class RobotModel:
    def __init__(self, urdf_path, last_link_name, load_mesh,device):
        super().__init__()
        self.device = device   
        self.urdf_path = urdf_path
        self.last_link_name = last_link_name
        with open(urdf_path, 'r') as urdf_file:
            urdf_data = urdf_file.read().encode()
        self.robot = pk.build_serial_chain_from_urdf(urdf_data, last_link_name).to(dtype=torch.float32, device=device)
        joint_lim = torch.tensor(self.robot.get_joint_limits(), device=device)
        self.theta_min = joint_lim[0].to(device)
        self.theta_max = joint_lim[1].to(device) 
        self.n_joints = len(self.theta_min)
        if load_mesh:
            self.mesh_path = os.path.join(os.path.dirname(urdf_path), 'meshes/visual/*.obj')
            self.meshes = self.load_meshes()
    
    def forward_kinematics(self, q):
        ret = self.robot.forward_kinematics(q, end_only=False)
        transformations = {}
        for k in ret.keys():
            trans_mat = ret[k].get_matrix()
            transformations[k] = trans_mat
        return transformations

    def forward_kinematics_eef(self, q):
        eef = self.forward_kinematics(q)[self.last_link_name]
        eef_pos, eef_R = eef[:, :3, 3], eef[:, :3, :3]
        eef_quat = utils.matrix_to_quaternion(eef_R)
        # eef_quat = torch.nn.functional.normalize(eef_quat, p=2,dim=1)
        return eef_pos, eef_quat
    
    def effector_distance(self, q, target_pose):
        assert len(q) == len(target_pose)
        # position error: L2 norm distance 
        # orientation error: geodesic distance using logmap function
        pos, quat = self.forward_kinematics_eef(q)
        pos_error = torch.norm(pos-target_pose[:,:3],dim=1,keepdim=True)
        ori_error = torch.norm(utils.logmap_th(quat, target_pose[:, 3:]),dim=1,keepdim=True)
        return pos_error, ori_error
    
    def get_Jacobian(self, q):
        batch_size, num_joints = q.shape

        # Perform the forward kinematics
        pos, quat = self.forward_kinematics_eef(q)
        # Initialize Jacobians
        J_pos = torch.zeros(batch_size, 3, num_joints, device=self.device)
        J_quat = torch.zeros(batch_size, 3, num_joints, device=self.device)

        dq = torch.zeros_like(q)
        delta = 1e-2
        for j in range(num_joints):
            dq[:, j] = delta
            pos_dq, quat_dq = self.forward_kinematics_eef(q + dq)
            J_pos[:, :, j] = (pos_dq - pos) / delta
            J_quat[:, :, j] = utils.logmap_th(quat_dq, quat) / delta
            dq[:, j] = 0
        return J_pos, J_quat

    
    def theta2mesh(self, q):
        trans = self.forward_kinematics(q)
        robot_mesh = []
        for k in self.meshes.keys():
            if k !='finger':
                mesh = copy.deepcopy(self.meshes[k])
                vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
                vertices = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=self.device)], dim=-1).t()
                transformed_vertices = torch.matmul(trans['panda_'+k].squeeze(), vertices).t()[:, :3].detach().cpu().numpy()
                mesh.vertices = transformed_vertices
                robot_mesh.append(mesh)
        return robot_mesh
    
    def load_meshes(self):
        mesh_files = glob.glob(self.mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4].split('_')[0]
            mesh = trimesh.load(mesh_file, force='mesh')
            meshes[name] = mesh
        return meshes
    
    def limit_angles(self,q):
        q = q % (2*torch.pi)  # Wrap angles between 0 and 2*pi
        q[q > torch.pi] -= 2 * torch.pi  # Shift angles to -pi to pi range
        q = torch.clamp(q, self.theta_min, self.theta_max)
        return q
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rbt = RobotModel(
        urdf_path=os.path.join('urdf/panda_urdf/panda.urdf'),
        last_link_name="panda_hand",
        load_mesh=False,
        device=device
    )
    x = torch.tensor([[0.0,0.5,0.3],[0.3,0.8,1.1]]).to(device)
    q = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, torch.pi/4]).float().to(device).unsqueeze(0).expand(len(x),-1)
    
    # eef_pos,eef_quat,sphere_center = rbt.forward_kinematics_all(q)
    # print(rbt.effector_distance(q,x))
    print(rbt.forward_kinematics(q))