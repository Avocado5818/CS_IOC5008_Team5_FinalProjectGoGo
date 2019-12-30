# Final Project Dataset Preparing.

# import some common libraries
from matplotlib import pyplot as plt
from pycocotools import mask
from skimage import measure
import pandas as pd
import numpy as np
import json
import cv2
import os

# loading training dataset path and name
# just change TRAIN_PATH to TEST_PATH, also make test json file.
TRAIN_PATH = './final_dataset/stage1_train/'
train_ids = [x for x in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH+x)]

# make table
df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df['path'] = df.apply(lambda x:TRAIN_PATH +'/{}/images/{}.png'.format(x[0],x[0]), axis=1)
df['masks'] = df.apply(lambda x:TRAIN_PATH +'/{}/masks/'.format(x[0],x[0]), axis=1)
df.head()

# show an example
imid = df['id'][0]
image_path = df[df.id==imid].path.values[0]
print(image_path)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.show()

# coco format dict
coco_format = {
        "images": [],
        "categories": [{"supercategory": "nucleus", "id": 1, "name": "nucleus"}],
        "annotations": []}

# save image path
COCO_TRAIN_PATH = './final_dataset/coco/images_gray/'
annotation_index = 1

# make coco format image dict and write image
for i in range(len(df['id'])):
    imid = df['id'][i]                                   # training image name
    image_path = df[df.id==imid].path.values[0]          # training image path
    image = cv2.imread(image_path)                       # read image
    # image RGB to Gray, method = value method.
    image = image.max(axis=2)                            # RGB to Grayscale method
    cv2.imwrite(COCO_TRAIN_PATH+imid+'.png', image)      # image write
    images = {
            "height": image.shape[0],
            "width": image.shape[1],
            "id": i,
            "file_name": imid + ".png"}
    coco_format["images"].append(images)                 # write data to dict

# make coco format mask dict
anno_index = 1
for i in range(len(df['id'])):
    imid = df['id'][i]
    mask_dir = df[df.id==imid].masks.values[0]           # image mask path
    masks = os.listdir(mask_dir)                         # image every masks
    mimgs = []
    for mask_i in masks:
        mimg = cv2.imread(mask_dir + '/' + mask_i)       # read mask image
        mimg = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY)
        mimg[mimg==255] = 1                              # values change to 0 or 1
        mimgs.append(mimg)
    mimgs = np.array(mimgs)

    # transform mask to coco segmentation format
    for j in range(len(mimgs)):
        ground_truth_binary_mask = mimgs[j]
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)
        areas = ground_truth_area.tolist()
        # if mask area < 5, too small, ignore it!
        if(areas < 5):
            print('anno_index:', anno_index)
            continue
        # annotation dict
        annotation = {
                "segmentation": [],
                "area": ground_truth_area.tolist(),
                "iscrowd": 0,
                "image_id": i,
                "bbox": ground_truth_bounding_box.tolist(),
                "category_id": 1,
                "id": anno_index
            }
        anno_index += 1
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)
        coco_format["annotations"].append(annotation)    # write data to dict

# save json file
with open('./final_dataset/trainval.json', 'w') as f:
        json.dump(coco_format, f)
