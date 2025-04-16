from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

# アプリ起動時ステージデータ読み込み
with open("stage_data.json", encoding="utf-8") as f:
    STAGE_DATA = json.load(f)

# ルーティング
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/TurtleGraphics')
def turtle_graphics():
    stage = request.args.get('stage', '1')
    stage_info = STAGE_DATA.get(stage, {})
    block1_count = stage_info["blocks"].get("block1", 0)
    block2_count = stage_info["blocks"].get("block2", 0)
    block3_count = stage_info["blocks"].get("block3", 0)
    block4_count = stage_info["blocks"].get("block4", 0)
    block5_count = stage_info["blocks"].get("block5", 0)
    block6_count = stage_info["blocks"].get("block6", 0)
    block7_count = stage_info["blocks"].get("block7", 0)

    # ブロック総数を計算
    total_blocks = (
        block1_count +
        block2_count +
        block3_count +
        block4_count +
        block5_count +
        block6_count +
        block7_count
    )

    blocks_html = ""
    block_id = 1
    number_id = 1

    # block1 を生成
    for i in range(block1_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="fd">
                <p class="center"><input type="number" id = "number{number_id}" class="block_numbox" min="0" max="999" />歩進む<br /></p>
            </div>
        '''
        block_id += 1
        number_id += 1

    # block2 を生成
    for i in range(block2_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="rt">
                <p class="center"><input type="number" id = "number{number_id}" class="block_numbox" min="0" max="999" />度右に曲がる<br /></p>
            </div>
        '''
        block_id += 1
        number_id += 1

    # block3 を生成
    for i in range(block3_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="lt">
                <p class="center"><input type="number" id = "number{number_id}"  class="block_numbox" min="0" max="999" />度左に曲がる<br /></p>
            </div>
        '''
        block_id += 1
        number_id += 1

    # block4 を生成
    for i in range(block4_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="for1">
                <p class="center"><input type="number" id = "number{number_id}"  class="block_numbox" min="0" max="999" />回繰り返す<br /></p>
            </div>
        '''
        block_id += 1
        number_id += 1

    # block5 を生成
    for i in range(block5_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="for2">
                <p class="center">ここまで繰りかえす<br /></p>
            </div>
        '''
        block_id += 1
    
    # block6 を生成
    for i in range(block6_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="if1">
                <p class="center">
                    壁にぶつかったら<br />
                </p>
            </div>
        '''
        block_id += 1
    
    # block7 を生成
    for i in range(block7_count):
        blocks_html += f'''
            <div class="block" id="block{block_id}" data-type="if2">
                <p class="center">
                    ここまでIF<br />
                </p>
            </div>
        '''
        block_id += 1

    return render_template(
        'TurtleGraphics.html',
        blocks_html=blocks_html,
        total_blocks=total_blocks,  # ブロックの総数を追加
        title=stage_info.get("title", "課題"),
        image=stage_info.get("image", "canvas_default.png")
    )

@app.route('/match_images', methods=['POST'])
def match_images():
    try:
        data = request.json
        imgnum = int(data.get('imgnumData', '1'))
        imageData_base64 = data.get('imageData')

        img1 = cv2.imdecode(np.frombuffer(base64.b64decode(imageData_base64.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        if img1 is None:
            return jsonify({'matchingValue': 0.0})

        if 1 <= imgnum <= 20:
            img2 = cv2.imread(f'static/assets/canvas_{imgnum}.png')
        else:
            img2 = cv2.imread('static/assets/canvas_default.png')
        if img2 is None:
            return jsonify({'matchingValue': 0.0})

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return jsonify({'matchingValue': 0.0})

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        similarity = len(good_matches) / max(len(kp1), len(kp2))

        return jsonify({'matchingValue': similarity})
    except Exception as e:
        print("Error in /match_images:", e)
        return jsonify({'matchingValue': 0.0})

#直接実行時のみ起動
if __name__ == '__main__':
    app.run(debug=True)
