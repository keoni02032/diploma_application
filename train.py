import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import copy
import os.path as osp
import mmcv
import numpy as np
import mmdet
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets.pipelines.transforms import *
from clearml import Task

@DATASETS.register_module()
class ECO(CustomDataset):

    CLASSES = ('0.0', '1.0', '2.0', '3.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)
    
            label_prefix = self.img_prefix.replace('image_2', 'label_2')
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
    
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[1:5]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.int64),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.int64))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


def run(model_type):
    if model_type == "ssd":
        cfg = Config.fromfile("./configs/ssd/ssd300_coco.py")
    elif model_type == "yolox":
        cfg = Config.fromfile("./configs/yolox/yolox_l_8x8_300e_coco.py")
    elif model_type == "faster_rcnn":
        cfg = Config.fromfile("./configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py")
    if model_type == "yolox":
        cfg.data.workers_per_gpu = 2
    else:
        cfg.data.workers_per_gpu = 2

    cfg.dataset_type = "ECO"
    cfg.data_root = "d:/VKR/parser/mmdetection/kitti_tiny/"

    if model_type == "faster_rcnn":
        cfg.data.train.type = "ECO"
        cfg.data.train.data_root = "d:/VKR/parser/mmdetection/kitti_tiny/"
        cfg.data.train.ann_file = "train.txt"
        cfg.data.train.img_prefix = "training/image_2"
    else:
        cfg.data.train.dataset.type = "ECO"
        cfg.data.train.dataset.data_root = "d:/VKR/parser/mmdetection/kitti_tiny/"
        cfg.data.train.dataset.ann_file = "train.txt"
        cfg.data.train.dataset.img_prefix = "training/image_2"

    cfg.data.test.dataset.type = "ECO"
    cfg.data.test.dataset.data_root = "d:/VKR/parser/mmdetection/kitti_tiny/"
    cfg.data.test.dataset.ann_file = "val.txt"
    cfg.data.test.dataset.img_prefix = "training/image_2"

    cfg.data.val.type = "ECO"
    cfg.data.val.data_root = "d:/VKR/parser/mmdetection/kitti_tiny/"
    cfg.data.val.ann_file = "val.txt"
    cfg.data.val.img_prefix = "training/image_2"

    if model_type == "faster_rcnn":
        cfg.model.roi_head.bbox_head.num_classes = 17
    else:
        cfg.model.bbox_head.num_classes = 17

    if model_type == "ssd":
        cfg.load_from = "d:/VKR/parser/mmdetection/checkpoints/best_mAP_epoch_25.pth"
        cfg.work_dir = "d:/VKR/parser/mmdetection/ssd_p_14"
    elif model_type == "yolox":
        cfg.load_from = "d:/VKR/parser/mmdetection/checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
        cfg.work_dir = "d:/VKR/parser/mmdetection/yolox_p_22"
    elif model_type == "faster_rcnn":
        cfg.load_from = "d:/VKR/parser/mmdetection/checkpoints/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth"
        cfg.work_dir = "d:/VKR/parser/mmdetection/faster_rcnn_p_13"

    cfg.optimizer.momentum = 0.8
    cfg.workflow.append(('val', 1))
    cfg.optimizer.lr = 0.02 / 200
    cfg.lr_config.warmup_ratio = 1 / 1e10
    cfg.lr_config.warmup_iters=500
    cfg.log_config.interval = 50
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.metric = 'mAP'
    cfg.checkpoint_config.interval = 5
    cfg.evaluation.interval = 5
    cfg.runner.max_epochs = 15 # 2
    print(cfg.data.val.pipeline)

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    print(f'Config:\n{cfg.pretty_text}')
    print('Build dataset')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.test))
    else:
        print("end")
        return
    print(cfg.workflow)
    print('Build the detector')
    model = build_detector( cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    print('Add an attribute for visualization convenience')
    model.CLASSES = datasets[0].CLASSES
    print('Create work_dir')
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    print('Train detector')
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    task = Task.init("VKR", "ssd train_4")
    run("ssd")
    task.close()


    if log_dict['mode'] == 'train' and log_dict['iter'] == 300:
            ep = log_dict['epoch']
            Task.current_task().get_logger().report_scalar("graph train lr", "lr", iteration=ep, value=log_dict['lr'])
            Task.current_task().get_logger().report_scalar("graph train memory", "memory", iteration=ep, value=log_dict['memory'])
            Task.current_task().get_logger().report_scalar("graph train loss_cls", "loss_cls", iteration=ep, value=log_dict['loss_cls'])
            Task.current_task().get_logger().report_scalar("graph train loss_bbox", "loss_bbox", iteration=ep, value=log_dict['loss_bbox'])
            Task.current_task().get_logger().report_scalar("graph train loss_obj", "loss_obj", iteration=ep, value=log_dict['loss_obj'])
            Task.current_task().get_logger().report_scalar("graph train loss", "loss", iteration=ep, value=log_dict['loss'])
            Task.current_task().get_logger().report_scalar("graph train loss_rpn_cls", "loss_rpn_cls", iteration=ep, value=log_dict['loss_rpn_cls'])
            Task.current_task().get_logger().report_scalar("graph train loss_rpn_bbox", "loss_rpn_bbox", iteration=ep, value=log_dict['loss_rpn_bbox'])
            Task.current_task().get_logger().report_scalar("graph train acc", "acc", iteration=ep, value=log_dict['acc'])
            Task.current_task().get_logger().report_scalar("graph loss_rpn_cls", "train loss_rpn_cls", iteration=ep, value=log_dict['loss_rpn_cls'])
            Task.current_task().get_logger().report_scalar("graph loss_rpn_bbox", "train loss_rpn_bbox", iteration=ep, value=log_dict['loss_rpn_bbox'])
            Task.current_task().get_logger().report_scalar("graph acc", "train acc", iteration=ep, value=log_dict['acc'])
            Task.current_task().get_logger().report_scalar("graph loss_cls", "train loss_cls", iteration=ep, value=log_dict['loss_cls'])
            Task.current_task().get_logger().report_scalar("graphloss_bbox", "train loss_bbox", iteration=ep, value=log_dict['loss_bbox'])
            Task.current_task().get_logger().report_scalar("graph loss_obj", "train loss_obj", iteration=ep, value=log_dict['loss_obj'])
            Task.current_task().get_logger().report_scalar("graph loss", "train loss", iteration=ep, value=log_dict['loss'])
        elif log_dict['mode'] == 'val' and 'AP50' in log_dict:
            ep = log_dict['epoch']
            print(ep)
            Task.current_task().get_logger().report_scalar("graph val lr", "lr", iteration=ep, value=log_dict['lr'])
            Task.current_task().get_logger().report_scalar("graph val AP50", "AP50", iteration=ep, value=log_dict['AP50'])
            Task.current_task().get_logger().report_scalar("graph val mAP", "mAP", iteration=ep, value=log_dict['mAP'])
        elif log_dict['mode'] == 'val':
            ep = log_dict['epoch']
            Task.current_task().get_logger().report_scalar("graph val loss_cls", "loss_cls", iteration=ep, value=log_dict['loss_cls'])
            Task.current_task().get_logger().report_scalar("graph val loss_bbox", "loss_bbox", iteration=ep, value=log_dict['loss_bbox'])
            Task.current_task().get_logger().report_scalar("graph val loss_obj", "loss_obj", iteration=ep, value=log_dict['loss_obj'])
            Task.current_task().get_logger().report_scalar("graph val loss", "loss", iteration=ep, value=log_dict['loss'])
            Task.current_task().get_logger().report_scalar("graph val loss_rpn_cls", "loss_rpn_cls", iteration=ep, value=log_dict['loss_rpn_cls'])
            Task.current_task().get_logger().report_scalar("graph val loss_rpn_bbox", "loss_rpn_bbox", iteration=ep, value=log_dict['loss_rpn_bbox'])
            Task.current_task().get_logger().report_scalar("graph val acc", "acc", iteration=ep, value=log_dict['acc'])
            Task.current_task().get_logger().report_scalar("graph loss_rpn_cls", "val loss_rpn_cls", iteration=ep, value=log_dict['loss_rpn_cls'])
            Task.current_task().get_logger().report_scalar("graph loss_rpn_bbox", "val loss_rpn_bbox", iteration=ep, value=log_dict['loss_rpn_bbox'])
            Task.current_task().get_logger().report_scalar("graph acc", "val acc", iteration=ep, value=log_dict['acc'])
            Task.current_task().get_logger().report_scalar("graph loss_cls", "val loss_cls", iteration=ep, value=log_dict['loss_cls'])
            Task.current_task().get_logger().report_scalar("graph loss_bbox", "val loss_bbox", iteration=ep, value=log_dict['loss_bbox'])
            Task.current_task().get_logger().report_scalar("graph loss_obj", "val loss_obj", iteration=ep, value=log_dict['loss_obj'])
            Task.current_task().get_logger().report_scalar("graph loss", "val loss", iteration=ep, value=log_dict['loss'])