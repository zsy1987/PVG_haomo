#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import json
import os    
import torch
import torch.nn.functional as F
from utils.loss_utils import psnr, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight
from utils.general_utils import seed_everything, visualize_depth
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf
import cv2
EPS = 1e-5
import torchvision


from plyfile import PlyData, PlyElement
import numpy as np
import torch.nn as nn
import torch



def load_ply(path,max_sh_degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        _features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        _opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # time_duration=[-0.98,0.98]
        fused_times = torch.full((_xyz.shape[0], 1), 0., device="cuda") 
        scales_t = torch.full((_xyz.shape[0], 1), 9999999999999999999999., device="cuda") #β无穷大
        velocity = torch.zeros(_xyz.shape[0], 3, device="cuda")  # 全部初始化为0
        # velocity[:, 1] = 3
        # velocity[:, 0] = 30
        _t = nn.Parameter(fused_times.requires_grad_(True))
        _scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        _velocity = nn.Parameter(velocity.requires_grad_(True))

        return _xyz,_features_dc,_features_rest,_opacity,_scaling,_rotation,_t,_scaling_t,_velocity


def save_ply(points, filename):

    vertex = np.array([(points[i][0], points[i][1], points[i][2]) for i in range(len(points))],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_element = PlyElement.describe(vertex, 'vertex')

 
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(filename)





@torch.no_grad()
def evaluation(_xyz,_rotation,_scaling,iteration, scene : Scene, renderFunc, renderArgs, env_map=None):
    from lpipsPyTorch import lpips
    
    scale = scene.resolution_scales[0]
    if "kitti" in args.model_path:
        # follow NSG: https://github.com/princeton-computational-imaging/neural-scene-graphs/blob/8d3d9ce9064ded8231a1374c3866f004a4a281f8/data_loader/load_kitti.py#L766
        num = len(scene.getTrainCameras())//2
        eval_train_frame = num//5
        traincamera = sorted(scene.getTrainCameras(), key =lambda x: x.colmap_id)
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                            {'name': 'train', 'cameras': traincamera[:num][-eval_train_frame:]+traincamera[num:][-eval_train_frame:]})
    else:
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=scale)},
                        {'name': 'train', 'cameras': scene.getTrainCameras()})
    
    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            outdir = os.path.join(args.model_path, "eval", config['name'] + f"_{iteration}" + "_render")
            os.makedirs(outdir,exist_ok=True)


            thetas_t,T_trans_t=get_tracks()


            for idx, viewpoint in enumerate(tqdm(config['cameras'])):

                new_xyz,new_scaling,new_rotation = tracking(_xyz.clone(),_rotation.clone(),_scaling.clone(),idx,thetas_t,T_trans_t)
                scene.gaussians = replace_last_N(scene.gaussians,new_xyz,new_scaling,new_rotation)
                # 随时间t动态改变τ，以实现改变振动周期的效果
                scene.gaussians._t[-_xyz.shape[0]:] = torch.full((_xyz.shape[0], 1),  19/20* (-0.98+idx*0.02)+0.8, device="cuda")

                xyz = scene.gaussians.get_xyz_SHM(viewpoint.timestamp)
                
                z=xyz[:,2]
                with open('tensor_data.txt', 'w') as f:
                    for item in z:
                        f.write(f"{item.item()}\n")

                if idx==2:
                    save_ply(xyz, '/data15/DISCOVER_winter2024/zhengj2401/PVG/point_cloud.ply')
                    exit(0)
              
                render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, env_map=env_map)
                image  = torch.clamp(render_pkg["render"], 0.0, 1.0)
          
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                depth = render_pkg['depth']
                alpha = render_pkg['alpha']
                sky_depth = 900
                depth = depth / alpha.clamp_min(EPS)
                if env_map is not None:
                    if args.depth_blend_mode == 0:  # harmonic mean
                        depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
                    elif args.depth_blend_mode == 1:
                        depth = alpha * depth + (1 - alpha) * sky_depth
            
                depth = visualize_depth(depth)
                alpha = alpha.repeat(3, 1, 1)

                grid = [gt_image, image, alpha, depth]
                grid = make_grid(grid, nrow=2)

                save_image(image, os.path.join(outdir, f"{viewpoint.colmap_id:03d}.png"))

            #     l1_test += F.l1_loss(image, gt_image).double()
            #     psnr_test += psnr(image, gt_image).double()
            #     ssim_test += ssim(image, gt_image).double()
            #     lpips_test += lpips(image, gt_image, net_type='vgg').double()  # very slow

            # psnr_test /= len(config['cameras'])
            # l1_test /= len(config['cameras'])
            # ssim_test /= len(config['cameras'])
            # lpips_test /= len(config['cameras'])

            # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
            # with open(os.path.join(outdir, "metrics.json"), "w") as f:
            #     json.dump({"split": config['name'], "iteration": iteration, "psnr": psnr_test.item(), "ssim": ssim_test.item(), "lpips": lpips_test.item()}, f)




# 转为四元数
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# 四元数相乘
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(1)
    w2, x2, y2, z2 = q2.unbind(1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)

# 替换高斯
def replace_last_N(gaussians,new_xyz,new_scaling,new_rotation):
    N=new_xyz.shape[0]
    gaussians._xyz[-N:]=new_xyz.clone()
    gaussians._scaling[-N:]=new_scaling.clone()
    gaussians._rotation[-N:]=new_rotation.clone()
    return gaussians


def get_tracks():
    
    start_aabb = np.array([-2,0,0.05])
    end_aabb = np.array([4,-0.6,0.05])
    aabb_4 = np.array([15,-0.6,0.05])

    start_yaw = np.array([0])
    mid_yaw = np.array([np.pi/14])
    end_yaw = np.array([0])
    yaw_4 = np.array([0])

    linspace = np.linspace(
        start_aabb , end_aabb , 30
    ) # 30x3

    linspace2 = np.linspace(
        end_aabb , aabb_4 , 69
    ) # 30x3

    yaw_1 = np.linspace(start_yaw , mid_yaw , 15) # 15x1
    yaw_2 = np.linspace(mid_yaw , end_yaw , 15) # 15x1
    yaw_3 = np.linspace(end_yaw , yaw_4 , 69)
    yaws = np.concatenate([yaw_1,yaw_2,yaw_3],axis=0)
    linspaces=np.concatenate([linspace,linspace2],axis=0)

    return yaws,linspaces

# 根据t更改高斯
def tracking(_xyz,_rotation,_scaling,idx,thetas_t,T_trans_t):

    # 初始旋转+缩放变换
    matrix = torch.tensor([
        [1, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=torch.float32).cuda()  
    theta = torch.tensor(90 * (3.141592653589793 / 180))  # 将角度转换为弧度
    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32).cuda()
    rotation_qvec = rotmat2qvec(np.array([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ]))
    rotate_quaternion = torch.tensor(rotation_qvec).unsqueeze(0).cuda().float()
    rotation_qvec2 = rotmat2qvec(np.array([
        [1, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]))
    rotate_quaternion2 = torch.tensor(rotation_qvec2).unsqueeze(0).cuda().float()
    _xyz = _xyz.float() @ matrix@  rotation_matrix 
    _rotation = quaternion_multiply(rotate_quaternion2,quaternion_multiply(rotate_quaternion,_rotation.float()))

    theta = torch.tensor(1/100 *3.1415926)

    rotation_matrix = torch.tensor([
        [1.0, 0.0, 0.0],
        [0,torch.cos(theta), -torch.sin(theta)],
        [0,torch.sin(theta), torch.cos(theta)],
    ], dtype=torch.float32).cuda()
    rotation_qvec = rotmat2qvec(np.array([
        [1.0, 0.0, 0.0],
        [0,torch.cos(theta), -torch.sin(theta)],
        [0,torch.sin(theta), torch.cos(theta)],
        
    ]))
    rotate_quaternion = torch.tensor(rotation_qvec).unsqueeze(0).cuda().float()
    _xyz = _xyz.float() @ rotation_matrix
    _rotation = quaternion_multiply(rotate_quaternion,_rotation.float())


    # 轨迹旋转
    theta2=torch.tensor(thetas_t[idx]).squeeze(0)#与t相关
    rotation_matrix = torch.tensor([
        [torch.cos(theta2), -torch.sin(theta2), 0.0],
        [torch.sin(theta2), torch.cos(theta2), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32).cuda()
    rotation_qvec = rotmat2qvec(np.array([
        [torch.cos(theta2), -torch.sin(theta2), 0.0],
        [torch.sin(theta2), torch.cos(theta2), 0.0],
        [0.0, 0.0, 1.0]
    ]))
    rotate_quaternion = torch.tensor(rotation_qvec).unsqueeze(0).cuda().float()
    _xyz = _xyz.float() @ rotation_matrix
    _rotation = quaternion_multiply(rotate_quaternion,_rotation.float())


    # 初始平移变换
    T_int = torch.tensor([-2,0,0.05]).cuda()
    # 轨迹平移
    T_trans= torch.tensor(T_trans_t[idx]).cuda()#与t相关
    _xyz=_xyz+T_int
    _xyz=_xyz+T_trans

    scale=8
    _xyz = _xyz/scale
    _scaling = torch.log(torch.exp(_scaling)/scale)

    return _xyz,_scaling,_rotation

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()
    
    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)
    
    seed_everything(args.seed)

    sep_path = os.path.join(args.model_path, 'separation')
    os.makedirs(sep_path, exist_ok=True)
    
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    
    # if args.env_map_res > 0:
    #     env_map = EnvLight(resolution=args.env_map_res).cuda()
    #     env_map.training_setup(args)
    # else:
    env_map = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    # import ipdb 
    # ipdb.set_trace()
    gaussians.restore(model_params, args)
    
    # -----车辆高斯初始化-----
    _xyz,_features_dc,_features_rest,_opacity,_scaling,_rotation,_t,_scaling_t,_velocity = load_ply('/data15/DISCOVER_winter2024/zhengj2401/jeep-wrangler-rubicon-recon-jk-2017/point_cloud/iteration_30000/point_cloud.ply',gaussians.max_sh_degree)


    gaussians.densification_postfix(_xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation,
                                   _t, _scaling_t, _velocity)
  
    # gaussians._xyz=gaussians._xyz
    # gaussians._scaling = torch.log((torch.exp(gaussians._scaling)))
    

    # if env_map is not None:
    #     env_checkpoint = os.path.join(os.path.dirname(checkpoint), 
    #                                 os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
    #     (light_params, _) = torch.load(env_checkpoint)
    #     env_map.restore(light_params)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    evaluation(_xyz,_rotation,_scaling,first_iter, scene, render, (args, background), env_map=env_map)

    print("Evaluation complete.")