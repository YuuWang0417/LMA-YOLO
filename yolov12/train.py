import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", message="FlashAttention is not available")

def main():
    # 載入論文模型
    model = YOLO("yolov12n_LMA-YOLO.yaml").load("yolo12n.pt")
    # model = YOLO("runs/detect/lung_baseline_v2/weights/best.pt")

    # 訓練 (論文參數)
    model.train(
        data="../aortic_valve_colab.yaml",
        epochs=100,
        imgsz=640,  # 主動脈瓣高解析
        batch=16,
        optimizer='SGD',
        workers=8,
        lr0=0.001,
        weight_decay=0.0005,
        momentum=0.937,
        device=0,
        # resume=True,
        # amp=True,
        # cache='disk', 
        name="pool_size_14",

        # mosaic=0.0,
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # scale=0.0,
        # translate=0.0,
        # fliplr=0.5,  
    )

if __name__ == "__main__":
    main()




# def main():
#     model = YOLO("runs/detect/train2/weights/best.pt")

#     #stage1
#     model.train(
#         data="D:/yolo_dataset_seg/dataset.yaml",
#         epochs=30,
#         patience=10,
#         imgsz=640,
#         batch=16,
#         optimizer="SGD",
#         workers=6,
#         lr0=0.0002,
#         weight_decay=0.0005,
#         momentum=0.937,
#         device=0,
#         name="train2_finetune_v2"
#     )

    #stage2
    # model = YOLO("runs/detect/train2_finetune_v2/weights/best.pt")
    # model.train(
    #     data="../aortic_valve_colab.yaml",
    #     epochs=20,
    #     patience=10,
    #     imgsz=640,
    #     batch=16,
    #     optimizer="SGD",
    #     workers=6,
    #     lr0=0.0001,
    #     weight_decay=0.0005,
    #     momentum=0.937,
    #     device=0,
    #     name="train2_finetune_v2"
    # )

# if __name__ == "__main__":
#     main()