# Periodic Vibration Gaussian for haomo
### [[Project]](https://fudan-zvg.github.io/PVG) [[Paper]](https://arxiv.org/abs/2311.18561) 

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