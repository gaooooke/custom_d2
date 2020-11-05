
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import custom_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode


def Predict():
    custom_metadata = MetadataCatalog.get("custom")
    # MetadataCatalog.get("custom").thing_classes= ["clothes"]
    cfg = get_cfg()
    cfg.merge_from_file(
        # "../detectron2-master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        "/media/ps/D/TeamWork/chengk/detectron2-master/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"
    )
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_het.pth")
    cfg.MODEL.WEIGHTS = "/media/ps/D/TeamWork/chengk/.custom_d2/output/model_final_het.pth"
    print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    data_f = '/media/ps/D/TeamWork/chengk/.custom_d2/pic/112.jpg'
    im = cv2.imread(data_f)
    outputs = predictor(im)
    print(outputs)
    # for k,v in outputs.items():
    #     print(k)
    v = Visualizer(im[:, :, ::-1],
                   metadata=custom_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE  # COCO : ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    Predict()