from fun import Tool, Camera, InternetSeletter
import gradio as gr
import os, json
from typing import Tuple, List, Dict, Any

# ==========================================
# 參數設定與物件初始化
# ==========================================

# N8N Webhook URL (自動化流程串接端點)
n8n = "https://damn-hack.app.n8n.cloud/webhook/b1b68338-d74a-4ffe-8559-65b3e6666158"
# 生成檔案的儲存路徑
save_dir = "./data"

# 初始化核心物件
# Camera: 負責攝影機串流與 YOLO 辨識 (0 代表預設攝影機)
camera = Camera(0, "./model/")
# Tool: 通用工具 (路徑處理、HTML生成)
tool = Tool()
# InternetSeletter: 負責與 N8N 溝通進行生成
internet = InternetSeletter(n8n, save_dir)

# 強制切換工作目錄至腳本所在位置，避免讀檔錯誤
tool.ChDir()

# 讀取下拉選單的選項資料
with open('DropdownData.json', 'r', encoding='utf-8') as dropdown_data:
    DropDownList = json.load(dropdown_data)

# ==========================================
# 輔助函式
# ==========================================

def UpdataInfomation(utensils_name: str = None) -> Tuple[str, Dict, List[str]]:
    """
    更新器物資訊卡片與狀態。
    
    Args:
        utensils_name (str, optional): 指定的器物名稱，若無則自動抓取當前辨識到的器物。

    Returns:
        Tuple[str, Dict, List[str]]: 
            - InfoCard: 更新後的 HTML 資訊卡字串。
            - Info: 器物詳細資訊字典 (存入 State)。
            - data: 包含器物 ID 的列表 (存入 State)。
    """
    # 獲取 HTML 卡片內容與資訊字典
    InfoCard, Info = tool.InfomationCard(camera.UtensilsID(utensils_name))
    # 包裝器物 ID 以存入 State
    data = [camera.UtensilsID(utensils_name)]
    return InfoCard, Info, data

# ==========================================
# Gradio 介面建構
# ==========================================

with gr.Blocks() as demo:
    # 建立狀態變數 (State)，用於在不同元件間傳遞資料，不會顯示在畫面上
    save_info = gr.State([])       # 儲存當前器物的詳細資訊
    save_utensils_id = gr.State([]) # 儲存當前器物的 ID

    # 標題區塊
    gr.Markdown("# <div align=center>古 代 生 活 情 境 還 原 系 統</div>")

    # --- 上半部：即時影像與資訊卡 ---
    with gr.Row(equal_height=True, max_height=400):
        # 左側：攝影機畫面
        with gr.Column():
            # 串流影像元件 (streaming=True)
            YoloCamera = gr.Image(label="YOLO即時辨識畫面", height=350, streaming=True) 
            RetrieveUtensils = gr.Button("擷取物件照片") # 按下後鎖定當前辨識到的物件
        
        # 右側：器物資訊 HTML
        with gr.Column():
            HtmlInfocard = gr.HTML(elem_id="info_col", label="器物資訊卡", min_height=400)

    # --- 中間：參數設定區 ---
    with gr.Column():
        # 下拉選單：從 DropDownList 載入選項
        people      = gr.Dropdown(label="人物", choices=DropDownList[0], interactive=True, filterable=False)
        dress       = gr.Dropdown(label="著裝", choices=DropDownList[1], interactive=True, filterable=False)
        accessories = gr.Dropdown(label="配件", choices=DropDownList[2], interactive=True, filterable=False)
        style       = gr.Dropdown(label="照片風格", choices=DropDownList[3], interactive=True, filterable=False)
        context     = gr.Dropdown(label="情境", choices=DropDownList[4], interactive=True, filterable=False)
        
        # 生成按鈕
        TriggerReduction = gr.Button("進 行 生 成")

    # --- 下半部：生成結果展示 ---
    with gr.Row(equal_height=True):
        Img = gr.Image(label="情境還原圖", height=540)
        Video = gr.Video(label="情境影片", loop=True, autoplay=True)

    # ==========================================
    # 事件綁定 (Event Listeners)
    # ==========================================

    # 1. 頁面載入時，啟動攝影機串流傳輸到 YoloCamera 元件
    demo.load(camera.StreamVideo, None, YoloCamera)
    
    # 2. 點擊「擷取物件照片」按鈕：
    #    呼叫 UpdataInfomation -> 更新 HtmlInfocard, 儲存 save_info, 儲存 save_utensils_id
    RetrieveUtensils.click(UpdataInfomation, inputs=None, outputs=[HtmlInfocard, save_info, save_utensils_id])
    
    # 3. 頁面初始化時，先執行一次更新資訊卡 (顯示預設或未知狀態)
    demo.load(UpdataInfomation, None, [HtmlInfocard, save_info, save_utensils_id])

    # 4. 點擊「進行生成」按鈕：
    #    呼叫 internet.SendOutReduction -> 將所有下拉選單與器物資訊傳送至 N8N -> 回傳圖片與影片
    TriggerReduction.click(
        internet.SendOutReduction,
        inputs=[
            save_info,   # 當前器物資訊
            people,      # 人物
            dress,       # 著裝
            accessories, # 配件
            style,       # 風格
            context      # 情境
        ],
        outputs=[Img, Video] # 輸出至圖片與影片元件
    )

if __name__ == "__main__":
    # 啟動 Gradio 應用
    # max_size=5: 限制佇列最大長度
    # css: 載入外部樣式表
    demo.queue(max_size=5).launch(css=tool.GetStyle(os.path.abspath('./style.css')), debug=True)