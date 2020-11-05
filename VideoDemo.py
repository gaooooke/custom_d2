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
from imutils.video import VideoStream
import os


def Predict():

    cfg = get_cfg()
    cfg.merge_from_file(
        # "../detectron2-master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        "/media/ps/D/TeamWork/chengk/detectron2-master/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"
    )
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = "/media/ps/D/TeamWork/chengk/.custom_d2/output/model_final_het.pth"
    print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    return predictor

model = Predict()

vid = 'rtsp://admin:qwe,asd.@10.10.15.154:554/h264/ch1/main/av_stream'
vs = VideoStream(src=vid).start() 

if __name__ == "__main__":
    custom_metadata = MetadataCatalog.get("custom")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    while True:
        frame = vs.read()    
        if frame is not None:
            try:
                im = cv2.imread(frame)
                outputs = model(im)
                v = Visualizer(im[:, :, ::-1],
                            metadata=custom_metadata,
                            scale=0.8,
                            instance_mode=ColorMode.IMAGE  # COCO : ColorMode.IMAGE_BW
                            )
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = v.get_image()[:, :, ::-1]
                cv2.imshow('rr', img)
                cv2.waitKey(1)
            except Exception as e:
                print(e)
                cv2.imshow('rr', frame)
                cv2.waitKey(1)