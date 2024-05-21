import cv2
import os

# 图片文件夹路径
images_folder = '/data15/DISCOVER_winter2024/zhengj2401/PVG/eval_output/haomo_reconstruction/eval/train_40000_render'

# 保存视频的文件名
output_video = '/data15/DISCOVER_winter2024/zhengj2401/PVG/output_video.mp4'

# 视频帧率
fps = 20

# 获取图片列表
image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.png')]
image_files=sorted(image_files)
# 获取第一张图片的大小
img = cv2.imread(image_files[0])
height, width, _ = img.shape

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 遍历图片列表并将每张图片写入视频
k=0
for image_file in image_files:
    img = cv2.imread(image_file)
    video_writer.write(img)
    # k+=1
    # if k==43:
    #     break

# 释放资源
video_writer.release()
