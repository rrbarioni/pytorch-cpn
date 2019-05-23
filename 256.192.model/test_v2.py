import os
import sys
sys.path.insert(0, '..')

import argparse

import torch
import cv2
import json
import numpy as np

from test_config import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.osutils import mkdir_p, isfile, isdir, join
# from dataloader.mscocoMulti import MscocoMulti
from dataloader.mscocoMulti_with_rotation import MscocoMulti
from tqdm import tqdm

from load_model import load_flattened_model_val, load_model_val
from predict import Predict
from predict import PredictWithRotation

def main(args):
    # model = load_flattened_model_val(args.checkpoint, args.test)
    model = load_model_val(args.checkpoint, args.test)

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg, train=False),
        batch_size=args.batch*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print('testing...')
    full_result = []
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        # full_result += Predict.predict_val(model, inputs, meta)
        full_result += PredictWithRotation.predict_val(model, inputs, meta, 10)
        if i == 100:
            break

    result_path = args.result
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result.json')
    with open(result_file,'w') as wf:
        json.dump(full_result, wf)

    # evaluate on COCO
    eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')      
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str,
        metavar='PATH', help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='CPN256x192', type=str,
                        help='using which checkpoint to be tested \
                              (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result', type=str,
                        help='path to save save result (default: result)')
    main(parser.parse_args())