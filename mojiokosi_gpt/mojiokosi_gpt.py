import PySimpleGUI as sg
import os
import subprocess
import torch
import ffmpeg
from pathlib import Path
import pandas as pd
import logging
import chardet
import time
import openai
import sys
import requests
from dotenv import load_dotenv



# ログファイルをクリアする
log_file = 'C:/tool/mojiokosi/app.log'
if os.path.exists(log_file):
    open(log_file, 'w',encoding='utf-8').close()

# ロギングの設定
logging.basicConfig(
    filename='C:/tool/mojiokosi/app.log',               # ログを出力するファイル名
    level=logging.INFO,                                 # ログレベルはINFO以上
    format='%(asctime)s %(levelname)s :%(message)s',    # '日時 ログレベル :メッセージ' の形式で出力
    datefmt='%Y/%m/%d %H:%M:%S',                        # 日時の表示形式を指定
)

# 環境変数の読み込み
load_dotenv()

########################
###  文字起こし処理   ###
########################
def transcribe(input_path, output_path, use_gpu):
    # whisper large-v3を取得
    logging.info('transcribe_start')
    from faster_whisper import WhisperModel

    # 画面で選択されたモデルの読み込みを行う
    # GPUを使用するかどうかでmodelの設定を切り替える
    #sg.Print('use_gpu',use_gpu)
    if use_gpu:
        logging.info('transcribe_gpu_select')
        if selected_process == '文字起こし（精度優先）':
            logging.info('transcribe_gpu_select_precision')    
            model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        else:  # 文字起こし（スピード優先）の場合
            logging.info('transcribe_gpu_select_speed')
            model = WhisperModel(model_size_or_path="deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda", compute_type="float16")
    else:
        logging.info('transcribe_cpu_select')
        if selected_process == '文字起こし（精度優先）':
            logging.info('transcribe_cpu_select_precision')
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        else:  # 文字起こし（スピード優先）の場合
            logging.info('transcribe_cpu_select_speed')
            model = WhisperModel(model_size_or_path="deepdml/faster-whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")

    # 文字起こししたデータを text file を出力するための関数
    def create_srt_file(file_name, results, fast_whisper=True):
        """文字起こし結果をテキストファイルとして保存する内部関数"""
        start_time = time.time()  # 処理開始時間を記録

        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                for segment in results:
                    loop_start_time = time.time()  # ループ開始時間を記録

                    # テキストのみを書き込み（タイムスタンプなし）
                    text = segment.text.strip()
                    f.write(text)

            logging.info(f"Total execution time: {time.time() - start_time} seconds")  # 総実行時間をログ出力

        except Exception as e:
            logging.error(f"ファイル作成中にエラーが発生しました: {str(e)}")
            raise

    # 画面の入力ファイル選択で確定した入力ファイルをINPUTにして文字起こし処理を実行する
    logging.info('transcribe start_point')
    results, _ = model.transcribe(input_path, language='ja')
    logging.info('transcribe end_point')
    logging.info('results', results)

    # 文字起こし結果を画面の出力結果の出力先選択で確定したファイルに書き出す
    logging.info('create_srt_file start_point')
    create_srt_file(file_name=output_path, results=results, fast_whisper=True)
    logging.info('create_srt_file end_point')

    return output_path

########################
###  文章校正処理     ###
########################
def correct_text(input_path, output_path, retries=3):
    logging.info('correct_text start_point')

    # <<<<< Fastapiにリクエストして、OpenAIのAPIKEYを取得する関数 >>>>>
    api_url = "https://fastapi-keyget-4cdd2eb0f921.herokuapp.com/verify-authentication-key"   # HerokuのURL
    authentication_key = os.getenv("Authentication_key")
    response = requests.post(api_url, json={"client_key": authentication_key})
    if response.status_code == 200:
        openai.api_key = response.json()["decrypted_api_key"]
    else:
        sg.popup("APIKEY取得の認証に失敗しました。",
        title="エラーメッセージ")
        return "Error: "  # 'Error:' を返す

    # 入力ファイルのエンコードを取得
    encoding = detect_encoding(input_path)

    # プロンプトファイルを読み込む
    with open(r"C:\tool\mojiokosi\correct_prompt.txt", "r",encoding='utf-8') as file:
        prompt_content = file.read()

    # 出力ファイルを開く
    with open(output_path, mode="w",encoding='utf-8') as f_out:
        # 入力ファイルを読み込む
        with open(input_path, mode="r", encoding=encoding) as f_in:
            text = f_in.read()

        # テキストを2048文字で分割して処理する
        start = 0
        while start < len(text):
            # 2048文字ずつに分割
            target_text = text[start:start+2048].strip()

            for i in range(retries):
                try:
                    messages=[
                        {"role": "system", "content": "あなたはプロの編集者です。"},
                        {"role": "user", "content": prompt_content.format(target_text=target_text)},
                    ]

                    # ChatGPT APIを使用して文章校正を行う
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        temperature = 0.0,
                        messages=messages
                    )

                    # 返信のみを出力
                    corrected_text = response.choices[0].message.content.strip()

                    # 文字列の"「"と"」"を取り除く
                    corrected_text = corrected_text.replace("「", "").replace("」", "")

                    # 出力ファイルに書き込む
                    f_out.write(corrected_text)

                    # 次の2048文字の開始位置を更新
                    start += 2048

                  
                    break  # 成功したらリトライループを抜ける (テストロジック確認時はコメントにする)

                except openai.ServiceUnavailableError:
                    if i < retries - 1:  # 最後のリトライ以外は再試行
                        logging.info('correct_text retry_start. Attempt number: %s', i+1)  # リトライ回数をログに出力
                        time.sleep(5)  # 5秒待つ
                        continue
                    else:
                        logging.info('correct_text retry exceeded. Returning error.')
                        return "RetryExceededError"  # リトライ上限超過エラーを返す

                except Exception as e:  # その他の例外を捕捉
                    logging.error("Unexpected error: %s", e)
                    return "Error: "  # 'Error:' を返す

    logging.info('correct_text end_point')
    return output_path

########################
###  文章要約処理     ###
########################
def summarize(input_path, output_path, retries=3):
    logging.info('summarize start_point')

    # <<<<< Fastapiにリクエストして、OpenAIのAPIKEYを取得する関数 >>>>>
    api_url = "https://fastapi-keyget-4cdd2eb0f921.herokuapp.com/verify-authentication-key"   # HerokuのURL
    authentication_key = os.getenv("Authentication_key")
    response = requests.post(api_url, json={"client_key": authentication_key})
    if response.status_code == 200:
        openai.api_key = response.json()["decrypted_api_key"]
    else:
        sg.popup("APIKEY取得の認証に失敗しました。", 
        title="エラーメッセージ")
        return "Error: "  # 'Error:' を返す

    # 入力ファイルのエンコードを取得
    encoding = detect_encoding(input_path)

    #プロンプトファイルを読み込む
    with open(r"C:\tool\mojiokosi\summarize_prompt.txt", "r",encoding='utf-8') as file:
        prompt_content = file.read()
    #  プロンプトをセットする

    # 出力ファイルを開く
    with open(output_path, mode="w",encoding='utf-8') as f_out:
        # 入力ファイルを読み込む
        with open(input_path, mode="r", encoding=encoding) as f_in:
            text = f_in.read()

        # テキストを2048文字で分割して処理する
        start = 0
        while start < len(text):
            # 2048文字ずつに分割
            target_text = text[start:start+2048].strip()

            for i in range(retries):
                try:
                    messages=[
                    {"role": "system", "content": "あなたはプロの編集者です。"},
                    {"role": "user", "content": prompt_content.format(target_text=target_text)},
                    ]

                    # ChatGPT APIを使用して文章要約を行う
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        temperature = 0.0,
                        messages=messages
                    )

                    #返信のみを出力
                    summarize_text = response.choices[0].message.content.strip()

                    # 文字列の"「"と"」"を取り除く
                    summarize_text = summarize_text.replace("「", "").replace("」", "")  

                    # 出力ファイルに書き込む
                    f_out.write(summarize_text)

                    # 次の2048文字の開始位置を更新
                    start += 2048
                    break  # 成功したらリトライループを抜ける (テストロジック確認時はコメントにする)  

                except openai.ServiceUnavailableError:
                    if i < retries - 1:  # 最後のリトライ以外は再試行
                        logging.info('summarize_text retry_start. Attempt number: %s', i+1)  # リトライ回数をログに出力
                        time.sleep(5)  # 5秒待つ
                        continue
                    else:
                        logging.info('summarize_text retry exceeded. Returning error.')
                        return "RetryExceededError"  # リトライ上限超過エラーを返す

                except Exception as e:  # その他の例外を捕捉
                    logging.error("Unexpected error: %s", e)
                    return "Error: "  # 'Error:' を返す

    logging.info('summarize end_point')
    return output_path

########################
###  文章句読点付与   ###
########################
def punctuate(input_path, output_path, retries=3):          #パンクチュエート
    logging.info('punctuate start_point')

    # <<<<< Fastapiにリクエストして、OpenAIのAPIKEYを取得する関数 >>>>>
    api_url = "https://fastapi-keyget-4cdd2eb0f921.herokuapp.com/verify-authentication-key"   # HerokuのURL
    authentication_key = os.getenv("Authentication_key")
    response = requests.post(api_url, json={"client_key": authentication_key})
    if response.status_code == 200:
        openai.api_key = response.json()["decrypted_api_key"]
    else:
        sg.popup("APIKEY取得の認証に失敗しました。", 
        title="エラーメッセージ")
        return "Error: "  # 'Error:' を返す

    # 入力ファイルのエンコードを取得
    encoding = detect_encoding(input_path)

    #プロンプトファイルを読み込む
    with open(r"C:\tool\mojiokosi\punctuate_prompt.txt", "r",encoding='utf-8') as file:
        prompt_content = file.read()
    #  プロンプトをセットする

    # 出力ファイルを開く
    with open(output_path, mode="w",encoding='utf-8') as f_out:
        # 入力ファイルを読み込む
        with open(input_path, mode="r", encoding=encoding) as f_in:
            text = f_in.read()

        # テキストを2048文字で分割して処理する
        start = 0
        while start < len(text):
            # 2048文字ずつに分割
            target_text = text[start:start+2048].strip()

            for i in range(retries):
                try:
                    messages=[
                    {"role": "system", "content": "あなたはプロの編集者です。"},
                    {"role": "user", "content": prompt_content.format(target_text=target_text)},
                    ]

                    # ChatGPT APIを使用して句読点付与を行う
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        temperature = 0.0,
                        messages=messages
                    )

                    #返信のみを出力
                    punctuate_text = response.choices[0].message.content.strip()

                    # 文字列の"「"と"」"を取り除く
                    punctuate_text = punctuate_text.replace("「", "").replace("」", "")  

                    # 出力ファイルに書き込む
                    f_out.write(punctuate_text)

                    # 次の2048文字の開始位置を更新
                    start += 2048

                    break  # 成功したらリトライループを抜ける (テストロジック確認時はコメントにする) 

                except openai.ServiceUnavailableError:
                    if i < retries - 1:  # 最後のリトライ以外は再試行
                        logging.info('punctuate_tex retry_start. Attempt number: %s', i+1)  # リトライ回数をログに出力
                        time.sleep(5)  # 5秒待つ
                        continue
                    else:
                        logging.info('punctuate_tex retry exceeded. Returning error.')
                        return "RetryExceededError"  # リトライ上限超過エラーを返す

                except Exception as e:  # その他の例外を捕捉
                    logging.error("Unexpected error: %s", e)
                    return "Error: "  # 'Error:' を返す

    logging.info('punctuate end_point')
    return output_path

####################################
###  入力ファイルの拡張子チェック   ###
####################################
def check_input_extension(input_path, allowed_extensions):
    ext = Path(input_path).suffix.lower()
    if ext not in allowed_extensions:
        sg.popup(f"入力ファイルに指定されたファイルの拡張子は処理対象外です。処理対象の拡張子：{', '.join(allowed_extensions)} です。", 
                   title="エラーメッセージ")
        return False
    return True

####################################
###  出力ファイルの拡張子チェック   ###
####################################
def check_output_extension(output_path, allowed_extension):
    ext = Path(output_path).suffix.lower()
    if ext != allowed_extension:
        sg.popup(f"出力ファイルに指定されたファイルの拡張子は処理対象外です。処理対象の拡張子：{allowed_extension} です。", 
                   title="エラーメッセージ")
        return False
    return True

####################################
###  ファイルのエンコード取得　　   ###
####################################
def detect_encoding(file_path):   #エンコード
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

############################
###  特定文字列変換処理　  ###
############################
def replace_text(input_path):
    # 置き換えたい文字列が格納されたExcelファイルを開く
    df = pd.read_excel('C:\\tool\\mojiokosi\\moji_replace.xlsx')

    # 入力されたファイルの内容を読み込む
    with open(input_path, 'r', encoding=detect_encoding(input_path)) as f:
        text = f.read()

    # Excelファイルの内容に従って文字列を置き換える
    for index, row in df.iterrows():
        text = text.replace(row['変換対象文字列'], row['変換後文字列'])

    # 置き換えた後の文字列を一時ファイルとして保存
    temp_path = input_path + "_temp"
    with open(temp_path, 'w', encoding='utf-8') as ftemp:
        ftemp.write(text)

    return temp_path

############################
###  サブプロセス起動処理  ###
############################
def start_subprocess():
    # サブプロセス(GIFプログラム)終了指示ファイルが存在する場合、削除しておく
    terminate_file_path = 'C:/tool/mojiokosi/terminate.txt'
    if os.path.exists(terminate_file_path):
        os.remove(terminate_file_path)

    p = subprocess.Popen(["C:/tool/mojiokosi/gif_program.exe"])            #←２ pyinstall

    return p, terminate_file_path

############################
###  サブプロセス停止処理  ###
############################
def terminate_subprocess(p, terminate_file_path):
    # サブプロセス(GIFプログラム)の停止
    p.terminate()

    # サブプロセスの停止処理(p.terminate)が空振りした時の対応
    # メインプロセスが終了したことを示すために、終了指示ファイルを作成する
    with open(terminate_file_path, 'w') as f:
        f.write('terminate')

#------------------------------【メインコントロール】---- @@@@ -----------------------------------------------------------------------
# GUI レイアウトの定義
sg.theme('LightBlue1')  # テーマを設定        #  BlueMono LightBlue1 LightBlue2 LightBlue3 
#sg.set_options(background_color='white', text_element_background_color='white', element_background_color='white')

layout = [
    [
        sg.Column([
            [sg.Text('１．処理選択 （以下のいずれかの処理を選択して下さい）', text_color='black')],
            [sg.Text('    ', text_color='black'),sg.Listbox(values=('文字起こし（精度優先）','文字起こし（スピード優先）', '文章校正', '文章要約', '文章句読点付与'), size=(40, 4), key='-PROC-')],
            [sg.Text('    ', text_color='black'),sg.Text('文字起こしの場合は、CPUかGPUのどちらかを選択して下さい', text_color='black')],
            [sg.Text('    ', text_color='black'),sg.Radio('CPU使用', "RADIO1", default=True, key='-CPU-'), sg.Radio('GPU使用', "RADIO1", default=False, key='-GPU-')],
            [sg.Text('２．入力ファイル選択', text_color='black')],
            [sg.Text('    ', text_color='black'),sg.In(size=(80, 1), key='-IN-TEXT-'), sg.FileBrowse('ファイル選択', key='-IN-')],
            [sg.Text('３．出力結果の出力先選択', text_color='black')],
            [sg.Text('    ', text_color='black'),sg.Radio('既存ファイルに上書き出力', "RADIO2", default=False, key='-EXISTING-'), 
             sg.Radio('新規ファイルに出力', "RADIO2", default=False, key='-NEW-')],
            [sg.Text('    ', text_color='black'),sg.In(size=(80, 1), key='-OUT-TEXT-'), sg.FileBrowse('ファイル選択', key='-OUT-')],
            [sg.Text('４．出力ファイル一覧', text_color='black')],
            [sg.Text('    ', text_color='black'),sg.Text('文字起こし出力：', text_color='black'),sg.Output(size=(80, 1), key='-OUTLIST1-')],
            [sg.Text('    ', text_color='black'),sg.Text('文章校正出力　：', text_color='black'),sg.Output(size=(80, 1), key='-OUTLIST2-')],
            [sg.Text('    ', text_color='black'),sg.Text('文章要約出力　：', text_color='black'),sg.Output(size=(80, 1), key='-OUTLIST3-')],
            [sg.Text('    ', text_color='black'),sg.Text('句読点付与出力：', text_color='black'),sg.Output(size=(80, 1), key='-OUTLIST4-')],
            [sg.Text('ステータス:', font=("Helvetica", 10, "bold"), text_color='black'),
             sg.Text("", key='-STATUS-', size=(70, 1), font=("Helvetica", 10, "bold"), text_color='black')],
        ]),
        sg.VerticalSeparator(), 
        sg.Column([
            [sg.Button('実行', size=(10,1), button_color=('white', 'green'))],
            [sg.Button('終了', size=(10,1), button_color=('white', 'green'))],
            [sg.Button('クリア', size=(10,1), button_color=('white', 'green'))],
        ]),
    ]
]

icon_path = "C:/tool/mojiokosi/favicon.ico"     # アイコンファイルへのパス
window = sg.Window('文字起こし / 文章校正 / 文章要約 スクリプト', layout,icon=icon_path)
#window = sg.Window('文字起こし / 文章校正 / 文章要約 スクリプト', layout)

output_file_dict = {
    '文字起こし': '',
    '文章校正': '',
    '文章要約': '',
    '文章句読点付与': '',
}

try:
    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, '終了'):
            break

        elif event == 'クリア':
            for key in ['-IN-TEXT-','-OUT-TEXT-','-OUTLIST1-','-OUTLIST2-','-OUTLIST3-','-OUTLIST4-',
                        '-STATUS-']:
                window[key].update('')

        elif event == '実行':
            if values['-PROC-']:
                selected_process = values['-PROC-'][0]

                # 入力ファイルと出力ファイルのパスを取得
                input_path = values['-IN-TEXT-']
                output_path = values['-OUT-TEXT-']

                # 入力ファイルの存在チェック
                if not os.path.exists(input_path):
                    sg.popup('指定された入力ファイルが存在しません。',title="エラー")
                    continue

                # 出力結果の出力先選択のラジオボタンがどちらも選択されていない場合のチェック
                if not values['-EXISTING-'] and not values['-NEW-']:
                    sg.popup('出力結果の出力先選択がされていません。'
                             '「既存ファイルに上書き出力」または「新規ファイルに出力」を選択してください。'
                             ,title="エラー")
                    continue

                # ラジオボタンが「'既存ファイルに上書き出力'」の場合のチェック
                if values['-EXISTING-']:
                    if not os.path.exists(output_path):
                        sg.popup('指定された出力ファイルが存在しません。',title="エラー")
                        continue

                # ラジオボタンが「'新規ファイルに出力'」の場合のチェック
                elif values['-NEW-']:
                    if os.path.exists(output_path):
                        if sg.popup_ok_cancel('指定された出力ファイルが存在します。上書きしても宜しいですか？'
                                              ,title="確認") == 'OK':
                            pass
                        else:
                            sg.popup('指定内容を修正して下さい。',title="連絡")
                            continue

                if selected_process in ['文字起こし（精度優先）', '文字起こし（スピード優先）']:
                    # 入力ファイルの拡張子を確認
                    if not check_input_extension(input_path, ['.m4a', '.mp3', '.webm', '.mp4', '.mpga', '.wav', '.mpeg']):
                        continue
                    # 出力ファイルの拡張子を確認
                    if not check_output_extension(output_path, '.txt'):
                        continue

                else:  # 文字起こし以外の場合
                    # 入力ファイルの拡張子を確認
                    if not check_input_extension(input_path, '.txt'):
                        continue
                    # 出力ファイルの拡張子を確認
                    if not check_output_extension(output_path, '.txt'):
                        continue

                if selected_process in ['文字起こし（精度優先）', '文字起こし（スピード優先）']:
                    # GUIから入力ファイル、出力ファ��ル、GPUを使用するかの情報を取得
                    input_path = values['-IN-TEXT-']
                    output_path = values['-OUT-TEXT-']
                    use_gpu = values['-GPU-']
                    if use_gpu:
                    # GPUが使用可能か確認
                        if torch.cuda.is_available():
                            pass
                        else:
                            sg.popup("お使いのPCの環境ではGPUは使用できませんのでCPUを指定して下さい",
                                    title="エラーメッセージ")
                            continue
                    proceed = sg.popup_ok_cancel("文字起こし処理開始します",title="確認")
                    if proceed == 'OK':
                        # ステータスッセージを更新します。
                        window['-STATUS-'].update('文字起こし処理を実行中です。', text_color='red')
                        sg.popup("文字起こし処理を開始しました。当該処理は時間が掛かります。",title="連絡")

                        # サブプロセス(GIFプログラム)の起動関数
                        p, terminate_file_path = start_subprocess()

                        output_file_dict['文字起こし'] = transcribe(input_path, output_path, use_gpu)
                        window['-OUTLIST1-'].update(output_file_dict['文字起こし'])

                        # ステータスメッセージを更新します。
                        window['-STATUS-'].update('文字起こし処理が完了しました。', text_color='blue')

                        # サブプロセス(GIFプログラム)の停止関数
                        terminate_subprocess(p, terminate_file_path)

                        sg.popup("＊＊＊文字起こし処理終了＊＊＊",title="連絡")
                        logging.info('MOJIOKOSI_ALLEND')

                    else:
                        continue

                elif selected_process == '文章校正':
                    proceed = sg.popup_ok_cancel("文章校正処理を開始します",title="確認")
                    if proceed == 'OK':
                        # ステータスメッセージを更新します。
                        window['-STATUS-'].update('文章校正処理を実行中です。', text_color='red')
                        sg.popup("文章校正処理を開始しました。",title="連絡")
                        #特定文字列の変換処理
                        temp_path = replace_text(input_path)

                        # サブプロセス(GIFプログラム)の起動関数
                        p, terminate_file_path = start_subprocess()

                        output_file_dict['文章校正'] = correct_text(temp_path, output_path, retries=3)

                        if output_file_dict['文章校正'] == "RetryExceededError":
                            sg.popup("OpenAiへのアクセスがリトライオーバーになりました。暫く時間をおいてやり直して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        if isinstance(output_file_dict['文章校正'], str) and output_file_dict['文章校正'].startswith("Error:"):
                            sg.popup("予期せぬエラーが発生しました。アプリ開発者に連絡して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        # 一時ファイルの削除
                        os.remove(temp_path) 
                        window['-OUTLIST2-'].update(output_file_dict['文章校正'])
                        # ステータスメッセージを更新します。
                        window['-STATUS-'].update('文章校正処理が完了しました。', text_color='blue')

                        # サブプロセス(GIFプログラム)の停止関数
                        terminate_subprocess(p, terminate_file_path)

                        sg.popup("＊＊＊文章校正処理終了＊＊＊",title="連絡")
                    else:
                        continue

                elif selected_process == '文章要約':
                    proceed = sg.popup_ok_cancel("文章要約処理を開始します",title="確認")
                    if proceed == 'OK':
                        window['-STATUS-'].update('文章要約処理を実行中です。', text_color='red')
                        sg.popup("文章要約処理を開始しました。",title="連絡")
                        #特定文字列の変換処理
                        temp_path = replace_text(input_path)

                        # サブプロセス(GIFプログラム)の起動関数
                        p, terminate_file_path = start_subprocess()

                        output_file_dict['文章要約'] = summarize(temp_path, output_path, retries=3)

                        if output_file_dict['文章要約'] == "RetryExceededError":
                            sg.popup("OpenAiへのアクセスがリトライオーバーになりました。暫く時間をおいてやり直して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        if isinstance(output_file_dict['文章要約'], str) and output_file_dict['文章要約'].startswith("Error:"):
                            sg.popup("予期せぬエラーが発生しました。アプリ開発者に連絡して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        # 一時ファイルの削除
                        os.remove(temp_path)
                        window['-OUTLIST3-'].update(output_file_dict['文章要約'])
                        # ステータスメッセージを更新します。
                        window['-STATUS-'].update('文章要約処理が完了しました。', text_color='blue')

                        # サブプロセス(GIFプログラム)の停止関数
                        terminate_subprocess(p, terminate_file_path)

                        sg.popup("＊＊＊文章要約処理終了＊＊＊",title="連絡")
                    else:
                        continue

                elif selected_process == '文章句読点付与':
                    proceed = sg.popup_ok_cancel("文章句読点付与処理を開始します",title="確認")
                    if proceed == 'OK':
                        window['-STATUS-'].update('文章句読点付与処理を実行中です。', text_color='red')
                        sg.popup("文章句読点付与処理を開始しました。",title="連絡")
                        #特定文字列の変換処理
                        temp_path = replace_text(input_path)

                        # サブプロセス(GIFプログラム)の起動関数
                        p, terminate_file_path = start_subprocess()

                        output_file_dict['文章句読点付与'] = punctuate(temp_path, output_path, retries=3)

                        if output_file_dict['文章句読点付与'] == "RetryExceededError":
                            sg.popup("OpenAiへのアクセスがリトライオーバーになりました。暫く時間をおいてやり直して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        if isinstance(output_file_dict['文章句読点付与'], str) and output_file_dict['文章句読点付与'].startswith("Error:"):
                            sg.popup("予期せぬエラーが発生しました。アプリ開発者に連絡して下さい。", title="エラー")
                            terminate_subprocess(p, terminate_file_path)  # サブプロセス(GIFプログラム)の停止関数
                            continue

                        # 一時ファイルの削除
                        os.remove(temp_path) 
                        window['-OUTLIST4-'].update(output_file_dict['文章句読点付与'])
                        # ステータスメッセージを更新します。
                        window['-STATUS-'].update('文章句読点付与処理が完了しました。', text_color='blue')

                        # サブプロセス(GIFプログラム)の停止関数
                        terminate_subprocess(p, terminate_file_path)

                        sg.popup("＊＊＊文章句読点付与処理終了＊＊＊",title="連絡")
                    else:
                        continue

            else:
                sg.popup('　処理を選択してください　',title="エラー")
                continue

except Exception as e:
    logging.exception(f'エラーが発生しました: {e}')
    raise

finally:
    window.close()
