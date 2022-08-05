## Bring Your Own Data

Our model currently only support 360-degree data. Fast-forward data is not supported for now.

Please install [COLMAP](https://colmap.github.io/) before preparing your own data. 

To begin with, organize your dataset as the following structure (see BYOD_example at [here](https://drive.google.com/drive/folders/1HzxaO9CcQOcUOp32xexVYFtsyKKULR7T?usp=sharing)):
```
data\
|-- images_unmasked\
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
|-- images_masks\
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
|-- (optional) sparse\
    |-- 0\
        |-- images.bin
        |-- points3D.bin
        |-- cameras.bin
```

the image masks should be B/W masks. If your images are collected from different places, it is strongly recommended to mask out the image backgrounds with white color.

It is optional, however highly recommended, to execute COLMAP's SfM algorithm on its GUI by yourself. If your images are collected from the Internet, it might be very hard for COLMAP to solve a valid registration, and the GUI can be quite helpful for tunning. 

It is also suggested to apply any advanced correspondence network, such as SuperGlue, to improve the quality of camera poses. See [here]( https://colmap.github.io/database.html#keypoints-and-descriptors).

After the dataset is prepared, run:
```
cd utils/data_preproccess
python 1_mask_image.py --scenedir <path of your dataset>
python 2_preprocess_llff.py --scenedir <path of your dataset>
```
and your data is ready to go.