# 📂 model/ - AI 辨識模型核心區

## 📝 資料夾概述

`model/` 是本專案的 **AI 推論 (Inference) 核心目錄**。它存放了經過訓練的 **TensorFlow SavedModel** 格式模型與對應的標籤文件。

當程式啟動時，`fun.py` 中的 `Camera` 類別會載入此目錄下的模型，將攝影機捕捉到的影像進行運算 (Image Classification)，判斷畫面中出現的是哪一種古代器物（例如「陶缽」或「圓腹罐」）。

## 🏗️ 必要檔案結構 (Directory Structure)

本專案採用 TensorFlow 標準的 **SavedModel** 格式，因此此目錄的內部結構**必須**嚴格遵守以下配置：

```text
model/
├── 📂 model.savedmodel/      # [重要] 這是一個「資料夾」，而非單一檔案
│   ├── 📄 saved_model.pb     # 模型架構定義檔 (Protocol Buffer)
│   └── 📂 variables/         # 模型權重參數資料夾
└── 📄 labels.txt             # 分類標籤對照表

```

### 詳細說明

1. **`model.savedmodel/` (資料夾)**
* **類型**：TensorFlow SavedModel 格式目錄。
* **用途**：包含完整的神經網路架構與訓練好的權重 (Weights)。
* **程式載入**：`fun.py` 使用 `tf.keras.layers.TFSMLayer` 直接指向此**資料夾路徑**來載入模型。
* **來源**：通常由 Google Teachable Machine 匯出或透過 TensorFlow 程式碼儲存 (`model.save()`) 而來。


2. **`labels.txt`**
* **類型**：純文字檔。
* **用途**：定義模型輸出節點 (Index) 與人類可讀名稱 (Class Name) 的映射關係。
* **格式**：每一行代表一個類別，順序必須與模型訓練時完全一致。
* **運作邏輯**：當模型預測結果為 `Index 0` 時，程式會讀取此檔案的第一行文字，並透過文字處理邏輯將其對應到 `UtensilsData.json` 中的 ID。



## ⚙️ 模型技術規格 (Technical Specs)

若您打算自行訓練或更換模型，新的模型必須符合以下輸入/輸出規格，否則程式可能會報錯或辨識準確度低落：

* **框架版本**：TensorFlow 2.x
* **模型格式**：SavedModel (非 Keras .h5)
* **輸入形狀 (Input Shape)**：`(224, 224, 3)`
* 寬度：224 px
* 高度：224 px
* 通道：3 (RGB 彩色)


* **預處理 (Preprocessing)**：
* 程式碼 (`fun.py`) 中已內建正規化處理：`(Pixel / 127.5) - 1`。
* 這意味著模型訓練時必須使用相同的正規化方式（將像數值縮放至 -1 到 1 之間）。



## 🔄 如何更新模型？

如果您訓練了新的影像分類模型（例如使用 Teachable Machine），請依照以下步驟更新：

1. **匯出模型**：選擇 "TensorFlow" -> "SavedModel" 格式下載。
2. **解壓縮**：您會得到一個包含 `saved_model.pb` 的資料夾。
3. **重新命名與覆蓋**：
* 將解壓後的模型資料夾重新命名為 `model.savedmodel` 並覆蓋原資料夾。
* 將新的 `labels.txt` 覆蓋原檔案。


4. **檢查標籤**：確保 `labels.txt` 中的類別順序與新模型一致，且名稱能被程式邏輯正確解析為器物 ID。