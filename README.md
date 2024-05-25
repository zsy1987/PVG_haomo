# Periodic Vibration Gaussian for haomo


## Get started
### Environment
```
# Clone the repo.
git@github.com:zsy1987/PVG_haomo.git
cd PVG_haomo

# Make a conda environment.
conda create --name pvg python=3.9
conda activate pvg

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

```

### Data preparation

#### Haomo dataset
you can find haomo dataset example at /mnt/ve_share3/share/air/HC04
### Running

```
# Training

sh train.sh  # Expect 1 hour of training on a single GPU 3090



# eval+car 

sh render_car.sh  # This will take a few minutes
```
训练完成后，可以用 cloudcompare 打开 /HC04/points3d.ply 以确定待插入车辆的大小和轨迹，
并在./PVG_haomo/evaluate.py中290-301行修改插入车辆的超参数
```
#-----------------超参数-----------------
# 这里输入包括，目标车辆长度，车辆轨迹的三个点，yaw角的插值变化
target_car_length = 0.18 # 在点云中确定目标场景车辆长度约 0.18
xyz_init=np.array([-0.3423,0.1286,0.05]) # 设置车辆初始所在位置
xyz_2=np.array([0.2622,-0.006,0.047]) # 设置车辆变道后所在位置
xyz_3=np.array([1.781824,-0.006,0.041]) # 设置车辆直行后最终位置

yaw_1 = np.array([0]) # 初始 yaw 角为0，车头超x轴正方向
yaw_2 = np.array([np.pi/14]) # 变道前半段 从 0 插值到 π/14
yaw_3 = np.array([0]) # 变道后半段 从 π/14 插值到 0
yaw_4 = np.array([0]) # 直行 yaw 角保持为 0 
#----------------------------------------
```