import os
import math
import argparse

import torch
import numpy as np

import network
import dataset
import misc_utils
from config import config

if_set_nms = False

def eval_all(args):
    # model_path
    saveDir = config.model_dir
    evalDir = config.eval_dir
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 
            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # get devices
    #str_devices = args.devices
    #devices = misc_utils.device_parser(str_devices)
    # load data
    #records = misc_utils.load_json_lines(config.eval_source)

    coco = coco.COCO(config.eval_source)
    records = coco.getImgIds()
    num_records = len(records)
    print('val image number: {}'.format(num_records))


    all_results = []
    inference(model_file, args.devices, records)


def inference(model_file, device, records):
    torch.set_default_tensor_type('torch.FloatTensor')
    net = network.Network()
    net.cuda(device)
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    gts = []
    pts = []
    for record in records:
        np.set_printoptions(precision=2, suppress=True)
        net.eval()
        image, gt_boxes, im_info, ID = get_data(record, device)
        gts.append(gt_boxes)
        pred_boxes = net(image, im_info)
        if if_set_nms:
            from set_nms_utils import set_cpu_nms
            n = pred_boxes.shape[0] // 2
            idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))
            keep = pred_boxes[:, -2] > 0.05
            pred_boxes = pred_boxes[keep]
            keep = set_cpu_nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]
        else:
            import det_tools_cuda as dtc
            nms = dtc.nms
            keep = nms(pred_boxes[:, :4], pred_boxes[:, 4], 0.5)
            pred_boxes = pred_boxes[keep]
            pred_boxes = np.array(pred_boxes)
            #keep = pred_boxes[:, -1] > 0.05
            #pred_boxes = pred_boxes[keep]
        pts.append(pred_boxes)
                #rois=misc_utils.boxes_dump(rois[:, 1:], True))
    score_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    max_precisions, best_thres = 0, 0
    for i, score_thres in enumerate(score_thresholds):
        precisions = []
        for gt, pred in zip(gts, pts):
            # load preds
            pred = pred[pred[:,4]>score_thres]
            pred = pred[:,:4]
        
            precision = calculate_image_precision(gt, pred, thresholds=iou_thresholds, form='pascal_voc')
            precisions.append(precision)
   
        precisions = np.mean(precisions)
    
        if precisions > max_precisions:
            max_precisions = precisions
            best_thres = score_thres
        
        item = 'score_thres@{:.2f}'.format(score_thres)
        eval_results[item] = '{:.4f}'.format(precisions)
        
    eval_results['best_thres']= '@{:.1f}, {:.4f}'.format(best_thres, max_precisions)
    print(eval_results)




def get_data(record, device):
    data = dataset.val_dataset(record)
    image, gt_boxes, ID = \
                data['data'], data['boxes'], data['ID']
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    transposed_img = np.ascontiguousarray(
        resized_img.transpose(2, 0, 1)[None, :, :, :],
        dtype=np.float32)
    im_info = np.array([height, width, scale, original_height, original_width],
                       dtype=np.float32)
    image = torch.Tensor(transposed_img).cuda(device)
    im_info = torch.Tensor(im_info[None, :]).cuda(device)
    return image, gt_boxes, im_info, ID

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_test()

