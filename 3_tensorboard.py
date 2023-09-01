from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#add_image：在tensorboard中增加照片
#add_scalar：在tensorboard中增加标量
#怎么使用tensorboard：运行脚本，会在logs下面生成一个文件，运行 tensorboard --logdir=logs 即可
writer = SummaryWriter("logs")
image_path = "data/train/ants_image/6240329_72c01e663e.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, 1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()