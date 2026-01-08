from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2, threading, time, requests, gradio
import os, base64, json, zipfile, io
from typing import Optional, Tuple, Generator

class Tool:
    """
    工具函數庫
    負責處理檔案路徑、讀取設定檔、圖片編碼以及生成資訊卡片 HTML。
    """
    def __init__(self):
        pass

    def ChDir(self) -> None:
        """
        強制將工作區位址 (Current Working Directory) 改至 Python 腳本所在位置。
        避免因執行路徑不同導致讀取不到相對路徑的檔案。
        """
        if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    def GetStyle(self, style_path: str) -> str:
        """
        獲取介面會使用到的 CSS 美化代碼。
        
        Args:
            style_path (str): 美化表檔案的相對路徑 (例如: "style.css")。
            
        Returns:
            str: 讀取到的 CSS 內容，若檔案不存在則回傳空字串。
        """
        if os.path.exists(style_path):
            with open(style_path, "r", encoding="utf-8") as style_code_file:
                return style_code_file.read()
        return ""
    
    def CompileImage(self, img_path: str) -> str:
        """
        讀取圖片並轉換為 Base64 編碼字串，用於 HTML 嵌入。
        若是圖片無法獲取，將回傳一個透明的 1x1 GIF 圖片代碼。

        Args:
            img_path (str): 圖片檔案路徑。

        Returns:
            str: 帶有 Data URI Scheme 的 Base64 字串 (例如: "data:image/jpeg;base64,...")。
        """
        if os.path.exists(img_path):
            with open(img_path, "rb") as utensils_image:
                utensils_image_code = base64.b64encode(utensils_image.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{utensils_image_code}"
        # 回傳透明 1x1 像素圖片作為 Fallback
        return "data:image/jpeg;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    
    def InfomationCard(self, utensils_code: str) -> Tuple[str, dict]: # type: ignore
        """
        根據器物代碼生成 HTML 資訊卡片。

        Args:
            utensils_code (str): 器物的唯一識別碼 (例如: "Uten01")。

        Returns:
            Tuple[str, dict]: 
                - 格式化後的 HTML 字串。
                - 該器物的詳細資訊字典。
                - 若發生錯誤則回傳錯誤訊息 HTML 字串。
        """
        if os.path.exists('UtensilsData.json') and os.path.exists('InfomationCard.html'):
            # 讀取器物資料庫
            with open('UtensilsData.json', 'r', encoding="utf-8") as Utensils_Data:
                UtensilsList = json.load(Utensils_Data)

            # 讀取 HTML 模板
            with open('InfomationCard.html', 'r', encoding="utf-8") as Card:
                InfoCard = Card.read()

            # 獲取特定器物資料
            UtensilsCodeInfo = UtensilsList.get(utensils_code)

            if not UtensilsCodeInfo:
                utensils_code = "Unknown"
            
            # 建立資訊字典，準備填入 HTML 模板
            Info = {
                "name": UtensilsCodeInfo.get("器物名稱", "Unknown"),
                "base64": self.CompileImage("./Img/" + UtensilsCodeInfo.get("器物名稱", "Unknown") + ".jpg")
            }
            Info.update(UtensilsCodeInfo)

            try:
                # 使用 format_map 將資料填入 HTML
                return InfoCard.format_map(Info), Info
            except KeyError:
                return '<h4 style="color: red;">錯誤！ HTML內部參數缺失！</h4>', {} # 補上空的 dict 以符合 Tuple 結構
        return '<h4 style="color: red;">缺少依賴檔案</h4>', {}

class Camera:
    """
    攝像頭控制與物件偵測類別。
    負責讀取攝影機畫面、載入 TensorFlow 模型並進行即時推論。
    """
    def __init__(self, camera_id, model_path="./"):
        """
        Args:
            camera_id (int/str): 攝影機索引 ID (通常為 0 或 1)。
            model_path (str): 模型檔案所在的目錄路徑。
        """
        self.cam = {}
        self.uten = {}
        self.tensorflow = {}

        # 初始化攝影機參數
        self.cam['id'] = camera_id
        self.cam['cap'] = None
        self.cam['frame'] = None
        self.cam['running'] = False
        self.cam['lock'] = threading.Lock() # 確保多執行緒讀寫 frame 時的安全

        # 初始化器物資料
        self.uten['id'] = "Unknown"
        try:
            with open('UtensilsData.json', 'r', encoding='utf-8') as f:
                self.uten['data'] = json.load(f)
        except:
            self.uten['data'] = {}
        
        # 設定模型路徑
        self.tensorflow['model_dir'] = f'{model_path}model.savedmodel'
        self.tensorflow['labels_path'] = f'{model_path}labels.txt'
    
    def _update(self) -> None:
        """
        [內部方法] 後台執行緒主迴圈。
        持續讀取畫面 -> 預處理 -> 模型推論 -> 更新辨識結果。
        """
        # 1. 載入標籤檔
        try:
            with open(self.tensorflow['labels_path'], 'r', encoding='utf-8') as text:
                class_names = [line.strip() for line in text.readlines() if line.strip()]
        except Exception as e:
            print(f"Unable to read tag file: {e}")
            return
        
        # 2. 載入 TensorFlow 模型
        try:
            sm_layer = tf.keras.layers.TFSMLayer(self.tensorflow['model_dir'], call_endpoint='serving_default')
            inputs = tf.keras.Input(shape=(224, 224, 3))
            outputs = sm_layer(inputs)
            model = tf.keras.Model(inputs, outputs)
            print('Model loaded successfully.')
        except Exception as e:
            print(f'Model loaded failed: {e}')
            return

        # 3. 影像處理迴圈
        while self.cam['running']:
            success, img = self.cam['cap'].read()
            if success:
                # 影像預處理：BGR轉RGB -> Resize -> 正規化
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                pil_img = ImageOps.fit(pil_img, (224, 224), Image.Resampling.LANCZOS)

                image_array = np.asarray(pil_img)
                normalized_image = (image_array.astype(np.float32) / 127.5) - 1 # 將像素值縮放至 -1 到 1 之間
                input_data = np.expand_dims(normalized_image, axis=0)

                # 模型預測
                predictions_dict = model.predict(input_data, verbose=0)
                # 兼容不同格式的輸出
                if isinstance(predictions_dict, dict):
                    prediction = list(predictions_dict.values())[0]
                else:
                    prediction = predictions_dict
                
                index = np.argmax(prediction)
                confidence = prediction[0][index]

                # 若信心水準 > 0.7 則視為有效辨識
                if confidence > 0.7:
                    current_name = class_names[index]
                    name = current_name.split() # 假設標籤格式為 "ID Name"
                    # 解析並更新當前偵測到的 ID
                    utenid = "Uten0" + name[1][0] if current_name != "Unknown" else "Unknown"
                    self.uten['id'] = utenid
                
                # 更新畫面緩存 (加上文字標註)
                with self.cam['lock']:
                    display_frame = img.copy()
                    cv2.putText(display_frame, f"{class_names[index]} {confidence:.2%}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.cam['frame'] = display_frame
            else:
                time.sleep(0.01)

    def Start(self) -> None:
        """
        啟動攝影機並開啟辨識執行緒。
        """
        if self.cam['cap'] is None or not self.cam['cap'].isOpened():
            self.cam['cap'] = cv2.VideoCapture(self.cam['id'], cv2.CAP_DSHOW)
            self.cam['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cam['cap'].set(cv2.CAP_PROP_BUFFERSIZE, 1) # 減少延遲
            self.cam['running'] = True
            threading.Thread(target=self._update, daemon=True).start()

    def GetFrame(self) -> Optional[np.ndarray]:
        """
        獲取當前最新的影像幀。
        使用 Lock 確保讀取時不會與寫入衝突。
        """
        with self.cam['lock']:
            return self.cam['frame'].copy() if self.cam['frame'] is not None else None
    
    def UtensilsID(self, utensils_name: str = None) -> str:
        """
        獲取或設定當前的器物 ID。
        
        Args:
            utensils_name (str, optional): 若提供名稱，則反查 ID 並設定為當前 ID。
        
        Returns:
            str: 當前的器物 ID。
        """
        if utensils_name is None:
            return self.uten['id']
        else:
            for uid, info in self.uten['data'].items():
                if info.get('器物名稱') == utensils_name:
                    self.uten['id'] = uid
                    return uid
            return self.uten['id']
    
    def StreamVideo(self) -> Generator[np.ndarray, None, None]:
        """
        Gradio 專用的影像生成器 (Generator)。
        持續產出影像幀供前端顯示。
        """
        self.Start()
        try:
            while self.cam['running']:
                frame = self.GetFrame()
                if frame is not None:
                    # Gradio 預設吃 RGB (但 cv2 讀取是 BGR，這裡再轉回去看似多餘，
                    # 依據 self._update 存的是 BGR，Gradio Image 元件通常吃 RGB，
                    # 這裡保留原程式碼邏輯: BGR -> 存入 frame -> 取出 -> 轉 BGR (如果原 frame 是 RGB) 或不動)
                    # *註：原程式碼 _update 中存入的是 img.copy() (BGR)，這裡又轉 RGB2BGR 
                    # 可能原意是要輸出給特定格式，保持原樣不動。
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    yield frame
                time.sleep(0.03) # 約 30 FPS
        except GeneratorExit:
            self.Stop()
            print("Stream stopped.")
    
    def Stop(self) -> None:
        """
        停止攝影機與釋放資源。
        """
        self.cam['running'] = False
        if self.cam.get('cap'):
            self.cam['cap'].release()
        cv2.destroyAllWindows()

class InternetSeletter:
    """
    網路選擇器/發送器 (Internet Selector)
    負責與 N8N 自動化平台溝通，發送 Prompt 並接收生成的圖片/影片。
    """
    def __init__(self, n8n: str, save_dir: str):
        """
        Args:
            n8n (str): N8N Webhook URL。
            save_dir (str): 生成檔案的下載儲存目錄。
        """
        self.N8N = n8n
        self.SaveDir = save_dir
        self.Tool = Tool()
    
    def N8nProcess(self, prompt_word: str, utensils_url: str) -> bool:
        """
        發送 POST 請求至 N8N Webhook。
        
        Args:
            prompt_word (str): 組合好的提示詞。
            utensils_url (str): 器物參考圖片的 URL。

        Returns:
            bool: 請求是否成功並解壓縮完成。
        """
        SendWebHook = {
            "prompt_word": prompt_word,
            "utensils_url": utensils_url
        }
        try:
            # 發送請求，設定超時為 300 秒
            response = requests.post(url=self.N8N, json=SendWebHook, timeout=300)
            
            if response.status_code == 200:
                # 確保目錄存在
                os.makedirs(self.SaveDir, exist_ok=True)
                # 處理回傳的 Zip 檔案
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(self.SaveDir)
            elif response.status_code == 404:
                gradio.Warning("哎呀！ AI 大腦拒絕訪問！ 請重新讓它感受到它被需要吧！")
                return False
        except zipfile.BadZipFile:
            print("Error, This file is not in zip format.")
            gradio.Warning("哎呀！ AI 大腦燒穿了！ 請重新發出請求吧！")
            return False
        except requests.ConnectionError:
            print("Error, Unable to connectt to n8n webhook.")
            gradio.Warning("哎呀！ AI 大腦失去聯繫了！ 請重新呼叫它吧！")
            return False
        return True
    
    def SendOutReduction(self, info_dict: dict, person: str, dress: str, accessories: str, style: str, context: str) -> Tuple[Optional[str], Optional[str]]:
        """
        整合使用者輸入與器物資訊，生成最終提示詞並呼叫 N8N 進行生成。

        Args:
            info_dict (dict): 器物詳細資訊字典。
            person (str): 人物描述。
            dress (str): 服裝描述。
            accessories (str): 配件描述。
            style (str): 風格描述。
            context (str): 情境/背景描述。

        Returns:
            Tuple[Optional[str], Optional[str]]: (圖片路徑, 影片路徑)，若失敗則回傳 (None, None)。
        """
        gradio.Info("AI 大腦風暴運轉中..., 請等待約 2~3 分鐘")
        
        # 從字典提取資訊，若無則使用預設值
        u_name = info_dict.get("name", "未知器物")
        u_desc = info_dict.get("器物說明", "空白說明")
        uten_url = info_dict.get("照片網址", "https://drive.google.com/file/d/1ESNAsHTwCp97vmF_D8RLOCbt9eQIUd2c/view?usp=drive_link")
        
        # 組合 Prompt
        prompt_word = f"請生成一張{person}穿著{dress}身上有帶著{accessories}, {context}, 器物為:{u_name}, 整張圖片請用{style}風格生成, 以下為'{u_name}'這個器物的詳細說明:{u_desc}"
        
        # 呼叫 N8N 處理
        result = self.N8nProcess(prompt_word, uten_url)
        time.sleep(1)
        
        if result:
            return "./data/image.png", "./data/video.mp4"
        else:
            return None, None