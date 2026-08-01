from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/area8/weights/best.pt")
    model.predict(
        source='../datasets/test/images',
        imgsz=640,
        conf=0.25,
        save=True,
        save_conf=True,
        save_txt=True,   # ← 儲存預測的 label txt，比對用
        project='runs/predict',
        name='area=8',
        device=0,
        workers=0,
    )
