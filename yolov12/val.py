import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import multiprocessing
from multiprocessing import freeze_support
from ultralytics import YOLO

if __name__ == "__main__":
    freeze_support() 
    
    model = YOLO(r"./runs/detect/proposed_stageIII/weights/best.pt")  # 或 yolov12n.pt
    results = model.val(
        data="../aortic_valve_colab.yaml",
        imgsz=640,
        batch=16,
        workers=0,
        project="runs/val",
        name="test"
    )
    print(f"mAP@50: {results.box.map50:.4f}")
    print(f"mAP@50-95: {results.box.map:.4f}")

