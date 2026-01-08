# 📄 main.py - 應用程式入口與介面主控檔

## 📝 檔案概述

`main.py` 是本專案的 **主要執行入口 (Entry Point)**。它使用 **Gradio** 框架建構網頁互動介面，負責整合後端邏輯 (`fun.py`)、即時影像串流、使用者操作事件以及與外部 AI (N8N) 的溝通。

## 🛠️ 依賴模組與初始化

此檔案依賴於專案中的自定義模組 `fun.py` 以及標準庫：

* **`fun.py`**: 提供核心功能（`Tool`, `Camera`, `InternetSeletter`）。
* **`gradio`**: 用於構建 Web UI。
* **`os`**, **`json`**: 處理檔案路徑與數據讀取。

### 初始化物件

在程式啟動時，會優先初始化以下核心物件：

1. **Camera**: 設定攝影機 ID (預設 `0`) 與 **SavedModel 模型資料夾路徑** (`./model/`)。
2. **Tool**: 通用工具庫。
3. **InternetSeletter**: 設定 N8N Webhook URL 與檔案儲存路徑 (`./data`)。

---

## ⚙️ 核心功能邏輯

### 1. 資料載入

程式啟動時會讀取 `DropdownData.json`，將其內容載入至記憶體中，用於填充介面上的下拉選單（人物、著裝、配件等）。

### 2. 輔助函式 `UpdataInfomation`

這是連接後端辨識與前端顯示的橋樑函式：

* **輸入**：可選的器物名稱。
* **動作**：呼叫 `camera.UtensilsID` 獲取當前辨識到的器物 ID，並透過 `tool.InfomationCard` 生成對應的 HTML。
* **輸出**：更新介面上的 **HTML 資訊卡** 並將器物資料存入 **Gradio State** (狀態變數)。

---

## 🖥️ 介面佈局 (UI Layout)

介面使用 `gr.Blocks` 進行區塊化排版，分為上、中、下三層：

### 1. 標題區

* 顯示 **"古 代 生 活 情 境 還 原 系 統"** 標題。

### 2. 即時監控與資訊區 (Top Row)

* **左側 (TensorflowCamera)**：
* 顯示攝影機的即時串流畫面 (`streaming=True`)。
* 包含一個 **「擷取物件照片」** 按鈕。


* **右側 (HtmlInfocard)**：
* 顯示動態生成的 HTML 器物資訊卡（包含圖片、年代、說明）。



### 3. 參數設定區 (Middle Column)

提供 5 個下拉選單供使用者自訂生成情境，資料來源為 `DropdownData.json`：

* `people` (人物)
* `dress` (著裝)
* `accessories` (配件)
* `style` (照片風格)
* `context` (情境)
* 以及一顆 **「進 行 生 成」** 按鈕。

### 4. 生成結果展示區 (Bottom Row)

* **Img**: 顯示 AI 生成的情境還原圖片。
* **Video**: 顯示 AI 生成的情境影片。

---

## 🔗 事件互動 (Event Listeners)

此檔案定義了三個主要的互動邏輯：

1. **啟動攝影機串流**
```python
demo.load(camera.StreamVideo, None, TensorflowCamera)

```


* 網頁載入時，自動啟動後端攝影機並將畫面傳送至前端。


2. **擷取器物資訊**
```python
RetrieveUtensils.click(UpdataInfomation, inputs=None, outputs=[HtmlInfocard, save_info, save_utensils_id])

```


* 當使用者點擊「擷取物件照片」時，鎖定當前辨識到的物體，更新右側資訊卡，並將資料暫存於 `gr.State`。


3. **觸發 AI 生成 (N8N)**
```python
TriggerReduction.click(internet.SendOutReduction, inputs=[...], outputs=[Img, Video])

```


* 將使用者選擇的所有參數 (下拉選單) 與暫存的器物資訊打包。
* 發送至 N8N Webhook。
* 接收回傳的圖片與影片路徑並顯示於介面。



---

## 🚀 啟動方式

確保所有依賴檔案與虛擬環境已就緒後，執行：

```bash
python main.py

```

程式將啟動 Gradio 本地伺服器（預設為 `http://127.0.0.1:7860`），並載入 `style.css` 進行介面美化。