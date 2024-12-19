import tkinter as tk
import os
from PIL import Image, ImageTk, ImageSequence

class AnimatedGIF(tk.Label):
    def __init__(self, master, path, *args, **kwargs):
        tk.Label.__init__(self, master, *args, **kwargs)
        # GIFのフレームをロード
        self.sequence = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open(path))]
        self.image_pointer = 0
        self.current_image = self.sequence[0]  # 初期フレームを設定
        self.configure(image=self.current_image)
        self.after(1000, self.animate)  # アニメーション開始
        self.after(1000, self.check_for_termination)  # 終了条件のチェック

    def animate(self):
        # 次のフレームに移動
        self.image_pointer += 1
        if self.image_pointer == len(self.sequence):
            self.image_pointer = 0  # 最初のフレームに戻る
        self.current_image = self.sequence[self.image_pointer]
        self.configure(image=self.current_image)
        self.after(1000, self.animate)  # 次のフレームを表示

    def check_for_termination(self):
        # 終了条件を確認
        if os.path.exists('C:/tool/mojiokosi/terminate.txt'):
            try:
                os.remove('C:/tool/mojiokosi/terminate.txt')  # 終了ファイルを削除
            except OSError:
                pass  # 既に削除されている場合は無視
            root.destroy()  # ウィンドウを閉じる
        else:
            self.after(1000, self.check_for_termination)  # 再チェック

# ウィンドウの設定
root = tk.Tk()
root.title('処理中です。お待ちください...')

# ウィンドウサイズ
window_width = 330
window_height = 65

# スクリーンサイズを取得
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# ウィンドウの表示位置を調整（画面中央から右上にずらす）
offset_right = 210
offset_top = -185
position_top = int(screen_height / 2 - window_height / 2 + offset_top)
position_right = int(screen_width / 2 - window_width / 2 + offset_right)

# ウィンドウのサイズと位置を設定
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# GIFラベルを作成して配置
label = AnimatedGIF(root, "C:/tool/mojiokosi/sozai_cman_jp_20230716084851.gif")
label.pack()

root.mainloop()
