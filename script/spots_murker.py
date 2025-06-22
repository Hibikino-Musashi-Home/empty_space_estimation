import openai
import base64
import os
import cv2
import re
import matplotlib.pyplot as plt
import re



# input_image_path = "./testfile/image2_1.jpg"  # 入力画像のパス
# output_image_path = "./testfile/output_image_with_numbers.jpg"
result_path = "./testfile/result_numbers.jpg"










   

class ImageChatBot:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.n=0
    
    def add_numbers_to_image(self, image_path, output_path):
        # 画像を読み込む
        img = cv2.imread(image_path)
    
        
        # 画像のサイズを取得
        height, width = img.shape[:2]
        
        # フォントの設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)  # 青色
        thickness = 2
        
        # 数字を配置する位置を計算
        num_rows = 8
        num_cols = 8
        cell_height = height // num_rows
        cell_width = width // num_cols
        
        number = 0
        self.number_position = []
        for row in range(num_rows):
            for col in range(num_cols):
                # 数字の位置を計算
                x = col * cell_width + cell_width // 2
                y = row * cell_height + cell_height // 2

                
                # 数字をテキストとして追加
                text = str(number)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                self.number_position.append((x, y))
                cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
                
                number += 1
        
        # print(self.number_position)
        
        # 結果を保存
        cv2.imwrite(output_path, img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img_rgb)
        # plt.axis('off')
        # plt.show()
        print(f"数字を追加した画像を {output_path} に保存しました。")
       

    def add_number_to_image_next(self, select_number, image_path, output_path):
        img = cv2.imread(image_path)
        center_x, center_y = self.number_position[int(select_number)]

        # 中心位置に赤い四角形を描く
        marker_color = (255, 0, 0)  # 赤色 (BGR)
        marker_size = 40  # 四角形のサイズ
        marker_thickness = 2
        # 四角形の左上と右下の座標を計算
        top_left = (center_x - marker_size//2, center_y - marker_size//2)
        bottom_right = (center_x + marker_size//2, center_y + marker_size//2)
        cv2.rectangle(img, top_left, bottom_right, marker_color, marker_thickness)

        # 結果を保存
        cv2.imwrite(output_path, img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img_rgb)
        # plt.axis('off')
        # plt.show()
        print(f"選択された数字：{select_number} ")
        print(f"マーカーを追加した画像を {output_path} に保存しました。")


    def encode_image(self, output_path):
        image_list = []
        for image_path in [output_path]:
            image_list.append(image_path)
        print(image_path)
        self.base64_image_list = []
        for image_path in (image_list):
            print(image_path)
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                self.base64_image_list.append(base64_image)
        return  




    def set_inital_prompt(self) -> None:
        self.messages = [
            {"role": "system", "content": "棚の中に物を優しく置く必要があります。どの場所に置けば物が棚から落ちないのかを考えてください。"},
            {"role": "system", "content": "ユーザーの好み：((果物の近くには果物))、((スナックの近くにはスナック))、((飲み物の近くには飲み物))、((食べ物の近くには食べ物))、((ボールの近くにはボール))、((文房具の近くには文房具))、((好みを満たしたうえで最も近い空きスペース))、((棚の中))、((棚板の上))。"},
            {"role": "system", "content": "スペースが狭すぎる部分の数字は選ばないでください。"},
            {"role": "system", "content": "出力の最後に推奨番号：〇のような形で数字を出力して終わってください．"},
 
           
                 
            
        
        ]


    def create_chat(self,select_number, image_path, output_path):
        if self.n==0:
            self.add_numbers_to_image(image_path, output_path)
        else:
            self.add_number_to_image_next(select_number, image_path, output_path)
            
            
        self.encode_image(output_path)
        
        self.set_inital_prompt()
      
        self.messages.append(
            {"role": "user","content":
               [
                    {"type": "text", "text": "棚の中にある物と空きスペースにある番号を教えてください.その番号の中から新しくリンゴを置く際に適した番号を一つ出力してください。また、棚の外にある数字は絶対に選ばないでください。"},
                    {"type": "image_url","image_url": {"url":  f"data:image/jpeg;base64,{self.base64_image_list[0]}",},
                    },
                ],
            },
        )

        
        
    


        
        response = openai.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=self.messages,
            temperature=0,
        )
        self.n += 1
        return response.choices[0].message.content

    # def draw_bounding_box(self, bbox_coordinates, output_path):
    #     """
    #     画像内にバウンディングボックスを描画する関数。
        
    #     Parameters:
    #     image_list (str): 入力画像のパス
    #     bbox_coordinates (tuple): バウンディングボックスの座標 (x1, y1, x2, y2)
    #     """
    #     # 画像の読み込み
    #     print(image_list[0])
    #     image = cv2.imread(image_outpath)
        
    #     # バウンディングボックスの座標
    #     x1, y1, x2, y2= map(int, bbox_coordinates)
    #     print(x1, y1, x2, y2)
    #     # バウンディングボックスを描画
    #     start_point1 = (x1, y1)
    #     end_point1 = (x2, y2)
    #     color = (255, 0, 0)  # 緑色
    #     thickness = 2

    #     cv2.rectangle(image, start_point1, end_point1, color, thickness)

    #     # start_point2 = (x3, y3)
    #     # end_point2 = (x4, y4)
    #     # color = (0, 255, 0)  # 緑色
    #     # thickness = 2

    #     # cv2.rectangle(image, start_point2, end_point2, color, thickness)
    #     # print(output_path)
    #     cv2.imwrite(output_path, image)
    #     print(f"{output_path}に写真を保存しました。")

    #     img = cv2.imread(output_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # 画像を表示
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()
    #     self.messages.append( {"role": "user", "content": "その場所にはスペースがありません。ほかの位置に置きなおしてください。"},)


    
# select_number = None
# output_path = "./spots/testfile/output.jpg"
# chat_bot = ImageChatBot()
# response_content = chat_bot.create_chat(select_number, input_image_path, output_image_path)
# print(response_content)
# numbers = re.findall(r'\d+', response_content)
# print(numbers[-1])
# select_number = numbers[-1]
# input_image_path = output_image_path
# response_content = chat_bot.create_chat(select_number, input_image_path, output_image_path)
# # print(response_content)
if __name__ == "__main__":
    TEST_RUNS = 1   # 全体を何回くり返すか
    for run in range(TEST_RUNS):
        print(f"\n=== Test Run {run+1}/{TEST_RUNS} ===")
        input_image_path  = "./testfile/img_shelf1.jpg"
        output_image_path = "./testfile/output_image_with_numbers.jpg"
        # 1) 新しいインスタンスで初期化
        chat_bot = ImageChatBot()
        select_number = None
      
        
        # 2) 20 回のシーケンスを実行
        for i in range(2):
           
            resp = chat_bot.create_chat(select_number, input_image_path, output_image_path)
            # print("Response:", resp)
            
            nums = re.findall(r"\d+", resp)
            select_number = nums[-1] if nums else None
            input_image_path = output_image_path

