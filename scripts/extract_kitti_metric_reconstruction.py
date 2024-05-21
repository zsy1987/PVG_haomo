import os
import numpy as np
import json

root = "eval_output/kitti_reconstruction"
scenes = ["0001", "0002", "0006"]

eval_dict = {
    "TRAIN": {"psnr": [], "ssim": [], "lpips": []},
}
for scene in scenes:
    eval_dir = os.path.join(root, scene, "eval")
    dirs = os.listdir(eval_dir)
    test_path = sorted([d for d in dirs if d.startswith("train")], key=lambda x: int(x.split("_")[1]))[-1]
    for name, path in [("TRAIN", test_path)]:
        psnrs = []
        ssims = []
        lpipss = []
        with open(os.path.join(eval_dir, path, "metrics.json"), "r") as f:
            data = json.load(f)
        eval_dict[name]["psnr"].append(data["psnr"])
        eval_dict[name]["ssim"].append(data["ssim"])
        eval_dict[name]["lpips"].append(data["lpips"])
        
print(f'TRAIN PSNR:{np.mean(eval_dict["TRAIN"]["psnr"]):.3f} SSIM:{np.mean(eval_dict["TRAIN"]["ssim"]):.3f} LPIPS:{np.mean(eval_dict["TRAIN"]["lpips"]):.3f}')
