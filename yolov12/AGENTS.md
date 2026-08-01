# 醫學影像 Detection & Segmentation 研究專案

## 專案目標

使用 YOLO 系列模型進行醫學影像的 object detection，搭配 segmentation 模型（SAM 或 Mask R-CNN）做病灶區域切割，目標在目標資料集上達到競爭力的 mAP 與 Dice coefficient。

## 資料夾結構

- `datasets/` — 所有資料（**唯讀，不可修改原始資料！**）
  - `datasets/train/` — 訓練集
  - `datasets/val/` — 驗證集
  - `datasets/test/` — 測試集
  - `datasets/aug_window/` — augmentation 變體：sliding window
  - `datasets/aug_scale/` — augmentation 變體：scale jitter
  - `datasets/aug_mosaic/` — augmentation 變體：mosaic
  - `datasets/aug_local/` — augmentation 變體：local transform
  - `datasets/aug_hsv/` — augmentation 變體：HSV color shift
  - `datasets/aug_crop_scale/` — augmentation 變體：crop + scale
  - `datasets/aug_contrast/` — augmentation 變體：contrast adjust
  - `datasets/aug_clahe/` — augmentation 變體：CLAHE
- `configs/` — 模型設定檔（.yaml）
- `scripts/` — 前處理、訓練、評估腳本
- `runs/detect/` — 所有訓練實驗輸出（**不納入 git**）
  - `runs/detect/proposed_finetune_v22/` — 當前主力實驗（最新）
    - `weights/best.pt` — 最佳 checkpoint
    - `weights/last.pt` — 最後一個 epoch 的 checkpoint
    - `results.csv` — 每個 epoch 的 metrics（mAP@0.5、mAP@0.5:0.95、loss）
    - `results.png` — training curve 視覺化
    - `confusion_matrix.png` — confusion matrix
    - `PR_curve.png` / `F1_curve.png` — precision-recall 曲線
    - `val_batch*.jpg` — validation batch 預測結果視覺化
    - `args.yaml` — 這次訓練的完整參數紀錄
  - 其他歷史實驗（命名規則：`{strategy}_{version}`）：
    - `train2_finetune_v22/`、`train2_finetune_v2/`
    - `mixed_finetune_v3/`、`mixed_finetune_v2/`、`mixed2/`
    - `cosine_finetune_v3/`、`cosine_finetune_v2/`、`cosine_only/`
    - `proposed_finetune_v2/`
    - `n8/`、`n2/`
- `notebooks/` — 探索性分析用的 Jupyter notebook
- `output/` — 最終結果與視覺化

## 工作流程

**當我說「幫我整理今天的實驗」時：**
1. 讀取 `runs/detect/proposed_finetune_v22/results.csv` 與 `args.yaml`
2. 整理關鍵指標（mAP@0.5、mAP@0.5:0.95、box loss、cls loss 曲線趨勢）
3. 與前一版（`proposed_finetune_v2/`）比較，列出進步或退步的地方
4. 列出這次跑完的 observation 和還沒解決的問題
5. 輸出至 `output/logs/YYYY-MM-DD-proposed_finetune_v22-summary.md`

**當我說「幫我 debug 這個錯誤」時：**
1. 先看 error message 的類型（RuntimeError / CUDA OOM / shape mismatch…）
2. 從最常見原因開始排查，不要一開始就往複雜方向想
3. 給出可以直接跑的修正方法，或指出要去查的具體方向

**當我說「幫我看這個 config」時：**
1. 確認模型架構與資料集設定是否匹配（classes 數量、input size、anchor 設定）
2. 指出可能影響效能的參數（learning rate、augmentation、batch size）
3. 給出建議調整方向，附上理由

**當我說「幫我準備報告/簡報」時：**
1. 先問：對象是指導教授、實驗室 meeting、還是投稿用
2. 確認要呈現哪幾個實驗結果
3. 整理成結構：問題定義 → 方法 → 實驗設定 → 結果與比較 → 結論與下一步

## 工具偏好

- 訓練框架：PyTorch、Ultralytics（YOLOv8/v11）
- 資料增強：Albumentations
- 實驗追蹤：WandB（或 TensorBoard）
- 視覺化：matplotlib、OpenCV
- 環境：conda 虛擬環境，CUDA 版本需注意相容性

## 硬性限制

> - **不得修改 `data/` 下的任何原始檔案**
> - **不得自動推送 git commit**（手動執行）
> - **不得在 notebook 裡直接做最終訓練**（notebook 只用來探索，正式訓練跑 script）
> - 評估結果必須附上 evaluation script 或明確說明計算方式，不接受估算值