# Final Project Testing code.

# import some common libraries
from skimage.morphology import label
from scipy import ndimage
import pandas as pd
import random
import os

# import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

# loading model, weights, dataset
register_coco_instances("test", {}, "./final_dataset/coco/test.json", "./final_dataset/coco/test_images")
nucleus_metadata = MetadataCatalog.get("nucleus")
dataset_dicts = DatasetCatalog.get("test")
cfg.MODEL.WEIGHTS = os.path.join("./output", "model_0014999.pth")

# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("test", )
predictor = DefaultPredictor(cfg)

# mask to RLE-code function
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    A = run_lengths
    return run_lengths

# Testing
# post-processing setting.
score_param = [0.8]
interarea_param = [0.1]
area_param = [25]

for score_thres, area_inter_thres, area_threshold in zip(score_param, interarea_param, area_param):
    output = []
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        # model output
        outputs = predictor(im)
        instances = outputs['instances']
        n = os.path.split(d["file_name"])[1]
        image_id = n[:-4]

        # initial variables
        j = 0
        all_mask = None
        all_mask_no_refine = None
        # if model no prediction in this image, out = ''
        if(len(instances) < 1):
            out = ''
            output.append([image_id, out])
        else:
            # model predictions >= 1
            instances.pred_masks = instances.pred_masks.detach().cpu().numpy()

            for i in range(len(instances.pred_masks)):
                # if confidence < threshold
                if(instances.scores[i] < score_thres):
                    j = j + 1
                    continue
                else:
                    # fill holes
                    mask_int = ndimage.morphology.binary_fill_holes(instances.pred_masks[i].copy()).astype(np.uint8)
                    # mask to [truth, false] value
                    mask = mask_int > 0
                    # original mask
                    mask_orig = instances.pred_masks[i] > 0

                    # init all_mask & all_mask_no_refine = False (size = mask), run only once.
                    if all_mask is None:
                        all_mask = mask.copy() # make same size
                        all_mask[:] = False    # setting all False
                        all_mask_no_refine = mask_orig.copy()
                        all_mask_no_refine[:] = False

                    # intersection mask and all_mask
                    intersection = mask & all_mask
                    # sum intersection area
                    area_inter = intersection.sum()
                    # if this mask intersect to all_mask, and > 0.3 * mask area, ignore it!
                    if area_inter > 0:
                        total_area = mask.sum()
                        if float(area_inter) / (float(total_area) + 0.00001) > area_inter_thres:
                            j = j + 1
                            continue

                    # no intersection area < threshold, ignore it!
                    mask = mask & ~all_mask
                    if mask.sum() < area_threshold:
                        j = j + 1
                        continue

                    # setting mask_int no mask area = 0
                    mask_int[~mask] = 0
                    # add this mask to all_masks
                    all_mask = mask | all_mask
                    all_mask_no_refine = all_mask_no_refine | mask_orig
                    # mask to [0, 1] value
                    m = mask_int * 1
                    # mask to RLE-code
                    out = rle_encoding(m)
                    s = str(out)
                    s = s.replace('[', '')
                    s = s.replace(']', '')
                    s = s.replace(',', '')
                    output.append([image_id, s])

            # prediction score_thres, area_inter_thres, area_threshold all no legal
            if(j == len(instances.pred_masks)):
                out = ''
                output.append([image_id, out])

    # write csv file.
    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    submission = submission[submission['EncodedPixels'] != 'nan']
    submission_filepath = os.path.join('./final_dataset/Result', 'submission_{}_{}_{}.csv'.format(score_thres, area_threshold, area_inter_thres))
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')

