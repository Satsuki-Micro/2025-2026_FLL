# 📂 專案檔案結構與用途說明

本專案為一個結合 **Tensorflow 物件偵測**、**Gradio 前端介面**與 **Generative AI (N8N)** 的「古代生活情境還原系統」。以下為各檔案與資料夾的詳細用途說明。

## 📁 目錄結構

```text
Project_Root/
├── 📂 data/                 # 生成結果暫存區
├── 📂 Img/                  # 器物原始圖片庫
├── 📂 model/                # AI 辨識模型存放區
├── 📄 DropdownData.json     # 介面下拉選單選項
├── 📄 UtensilsData.json     # 器物詳細資訊數據
├── 📄 InfomationCard.html   # 資訊卡 HTML 模板
├── 🎨 style.css             # 介面美化樣式表
├── 🐍 fun.py                # 後端邏輯與工具函式庫
└── 🐍 main.py               # 程式入口與 Gradio 介面

```

---

## 🐍 核心程式碼 (Python Scripts)

### [`main.py`](./introduce/python-main.md)

* **用途**：**主程式入口 (Entry Point)**。
* **功能**：
* 使用 `Gradio` 建構網頁互動介面。
* 負責整合畫面串流、按鈕事件（擷取、生成）與狀態管理。
* 讀取設定檔並初始化系統。



### [`fun.py`](./introduce/python-fun.md)

* **用途**：**後端邏輯與工具庫 (Backend Library)**。
* **包含類別**：
* `Tool`：處理檔案路徑、圖片 Base64 編碼、HTML 模板填入。
* `Camera`：控制攝影機串流、載入 TensorFlow 模型、執行 Tensorflow 物件偵測。
* `InternetSeletter`：負責與 N8N 自動化平台溝通，發送 Prompt 並下載生成的圖片與影片。



---

## 📂 資料夾 (Directories)

### [`data/`](./introduce/data.md)

* **用途**：**生成結果暫存區**。
* **說明**：存放從 AI (N8N) 回傳並解壓縮後的檔案，預計包含生成的 `image.png` 與 `video.mp4`。

### [`Img/`](./introduce/img.md)

* **用途**：**器物原始圖片庫**。
* **說明**：存放器物的靜態展示圖（如 `鼎.jpg`）。當系統辨識到物體時，會從此處讀取對應圖片並轉碼顯示於資訊卡上。

### [`model/`](./introduce/model.md)

* **用途**：**AI 辨識模型存放區**。
* **說明**：必須包含 TensorFlow 輸出的模型檔案：
* `model.savedmodel`：模型權重與結構。
* `labels.txt`：對應的分類標籤文件。



---

## 📄 設定與數據檔案 (Data & Config)

### [`UtensilsData.json`](./introduce/utensilsdata.md)

* **用途**：**器物百科資料庫**。
* **內容**：JSON 格式，儲存每個器物 ID（如 `Uten01`）的詳細資料，包含名稱、年代、時代與詳細說明文字。

### [`DropdownData.json`](./introduce/dropdowndata.md)

* **用途**：**介面選單設定檔**。
* **內容**：JSON 格式，定義前端介面五個下拉選單的選項內容（人物、著裝、配件、風格、情境）。

---

## 🎨 前端樣式與模板 (Frontend Assets)

### [`InfomationCard.html`](./introduce/infomationdata.md)

* **用途**：**資訊卡 HTML 模板**。
* **說明**：定義器物資訊卡的 HTML 結構。內含 `{name}`, `{base64}` 等佔位符，供 Python 程式動態填入資料。

### [`style.css`](./introduce/style.md)

* **用途**：**介面美化樣式表**。
* **說明**：定義 Gradio 介面元件與 HTML 資訊卡的視覺樣式（CSS），如排版、顏色與字體。
