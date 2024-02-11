import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils import img_to_array

DT_folder = 'D:/Dev/Python/DispositivosMoviles/DT_data'
images_increased = 5

try:
    os.mkdir(DT_folder)
except:
    print("e")

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

data_path = 'D:/Dev/Python/DispositivosMoviles/images/train/tb'
data_dir_list = os.listdir(data_path)

width_shape, height_shape = 224, 224

i = 0
num_images = 0
for image_file in data_dir_list:
    img_list = os.listdir(data_path)

    img_path = data_path + '/' + image_file
    print(img_path)
    imge = load_img(img_path)
    print(imge)
    img_array = img_to_array(imge)

    imge = cv2.resize(img_array ,(width_shape,height_shape), interpolation = cv2.INTER_AREA)
    x = imge/255
    x = np.expand_dims(x,axis=0)
    t = 1
    for output_batch in train_datagen.flow(x, batch_size=1):
        a = output_batch[0]
        imagen = output_batch[0, :, :]*255
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        file_name = "%i%i.jpg" % (i, t)
        file_path = os.path.join(DT_folder, file_name)
        cv2.imwrite(file_path, imgfinal)
        t += 1

        num_images += 1
        if t > images_increased:
            break

        i += 1

print("images generated", num_images)
