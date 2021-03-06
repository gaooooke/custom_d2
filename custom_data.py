from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import register_pascal_voc
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

# coco
# register_coco_instances("custom", {}, "./COCO/trainval.json", "./COCO/images")
# custom_metadata = MetadataCatalog.get("custom")

# voc

# CLASS_NAMES=('headset','mask','hands','gloves','shoes','nomask','noheadset',)
# CLASS_NAMES = ('fire',)
CLASS_NAMES = ("sclothes","aclothes","clothes","glasses","nohat","hair","nohair","helmets","vests")
# CLASS_NAMES = ("hat","person",)

SPLITS = [
        ("custom_train", "EHSData", "train"),
        ("custom_val", "EHSData", "val"),
        # ("voc_self_val", "VOC2007", "trainval"),
    ]

for name, dirname, split in SPLITS:
    year = 2007 if "2007" in name else 2012
    register_pascal_voc(name, os.path.join("./", dirname), split, year, class_names=CLASS_NAMES)
    MetadataCatalog.get(name).evaluator_type = "pascal_voc"
