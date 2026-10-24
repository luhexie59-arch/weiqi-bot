"""Local desktop UI. Tk work stays on the main thread; CV runs in a worker."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from scoring import count_area
from vision import recognize, read_image, grid_coordinates
from reporting import draw_board, export_result


class ImageCanvas(tk.Canvas):
    def __init__(self, parent, click=None, right_click=None):
        super().__init__(parent, background="#e7ebf0", highlightthickness=0)
        self.image_data = None
        self.photo = None
        self.scale = 1.
        self.offset = (0, 0)
        self.bind("<Configure>", lambda event: self.redraw())
        if click:
            self.bind("<Button-1>", lambda e: self._click(e, click))
        if right_click:
            self.bind("<Button-3>", lambda e: self._click(e, right_click))

    def _click(self, event, callback):
        if self.image_data is None:
            return
        x = (event.x - self.offset[0]) / self.scale
        y = (event.y - self.offset[1]) / self.scale
        h, w = self.image_data.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            callback(x, y)

    def show(self, bgr):
        self.image_data = bgr
        self.redraw()

    def redraw(self):
        self.delete("all")
        if self.image_data is None:
            self.create_text(max(10, self.winfo_width() // 2), max(10, self.winfo_height() // 2),
                             text="打开棋盘图片，开始识别", fill="#526076", font=("Microsoft YaHei UI", 18))
            return
        h, w = self.image_data.shape[:2]
        cw, ch = max(10, self.winfo_width()), max(10, self.winfo_height())
        self.scale = min((cw - 8) / w, (ch - 8) / h)
        width, height = max(1, round(w * self.scale)), max(1, round(h * self.scale))
        self.offset = ((cw - width) / 2, (ch - height) / 2)
        image = Image.fromarray(cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS))
        self.create_image(*self.offset, image=self.photo, anchor="nw")


class GoCounterApp:
    def __init__(self, root):
        self.root = root
        root.title("围棋计数 · 当前占地")
        root.geometry("1220x870")
        root.minsize(960, 680)
        self.image = None
        self.source = None
        self.result = None
        self.board = None
        self.neutral_points = set()
        self.history = []
        self.manual_points = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.size = tk.IntVar(value=19)
        self.sensitivity = tk.DoubleVar(value=1.)
        self.mode = tk.StringVar(value="循环")
        self.status = tk.StringVar(value="支持 9 / 13 / 19 路；请先选择路数，再打开图片。")
        self.black_text = tk.StringVar(value="黑方  —")
        self.white_text = tk.StringVar(value="白方  —")
        self.neutral_text = tk.StringVar(value="中立  —")
        self._build()
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Escape>", lambda e: self.cancel_manual())
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=(9, 7), font=("Microsoft YaHei UI", 10))
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        toolbar = ttk.Frame(self.root, padding=12)
        toolbar.pack(fill="x")
        self.actions = []
        for text, command in (("打开图片", self.open_image), ("自动识别", self.run_auto),
                              ("手动四角", self.start_manual), ("重新识别棋子", self.reclassify),
                              ("撤销", self.undo), ("导出结果", self.export), ("载入结果", self.load_result)):
            button = ttk.Button(toolbar, text=text, command=command)
            button.pack(side="left", padx=(0, 6))
            self.actions.append(button)
        controls = ttk.Frame(self.root, padding=(14, 0, 14, 9))
        controls.pack(fill="x")
        ttk.Label(controls, text="路数").pack(side="left")
        self.size_combo = ttk.Combobox(controls, textvariable=self.size, values=(9, 13, 19), state="readonly", width=4)
        self.size_combo.pack(side="left", padx=(6, 18))
        self.size_combo.bind("<<ComboboxSelected>>", self.size_changed)
        ttk.Label(controls, text="棋子灵敏度").pack(side="left")
        self.sensitivity_spin = ttk.Spinbox(controls, from_=.5, to=2., increment=.1,
                                          textvariable=self.sensitivity, width=5)
        self.sensitivity_spin.pack(side="left", padx=(6, 18))
        ttk.Label(controls, text="左键修改").pack(side="left")
        ttk.Combobox(controls, textvariable=self.mode, values=("循环", "黑子", "白子", "空点"),
                     state="readonly", width=6).pack(side="left", padx=6)
        ttk.Label(controls, text="右键空点：整块标中立 / 取消；修改棋子后中立标记会清空。").pack(side="left", padx=10)
        scores = tk.Frame(self.root, bg="#162332", padx=20, pady=13)
        scores.pack(fill="x", padx=12)
        for variable, colour in ((self.black_text, "#6ed7ef"), (self.white_text, "#ffa7d2"),
                                 (self.neutral_text, "#d5dce6")):
            tk.Label(scores, textvariable=variable, bg="#162332", fg=colour,
                     font=("Microsoft YaHei UI", 16), anchor="w").pack(side="left", expand=True, fill="x")
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=10)
        self.original_canvas = ImageCanvas(self.tabs, self.source_click)
        self.photo_canvas = ImageCanvas(self.tabs, self.board_click, self.neutral_click)
        self.board_canvas = ImageCanvas(self.tabs, self.board_click, self.neutral_click)
        for canvas, label in ((self.original_canvas, "原图 / 四角标定"), (self.photo_canvas, "校正图 / 核对识别"),
                              (self.board_canvas, "棋盘 / 占地")):
            self.tabs.add(canvas, text=label)
        ttk.Label(self.root, text="青色方点＝黑方空地   粉色方点＝白方空地   灰色＝中立   橙框＝建议人工核对；空点→黑→白→空。",
                  padding=(14, 2)).pack(fill="x")
        ttk.Label(self.root, textvariable=self.status, wraplength=1150, padding=(14, 6)).pack(fill="x")

    def _busy(self, busy):
        for button in self.actions:
            button.configure(state="disabled" if busy else "normal")
        self.size_combo.configure(state="disabled" if busy else "readonly")
        self.sensitivity_spin.configure(state="disabled" if busy else "normal")

    def _submit(self, corners=None):
        if self.image is None or self.future:
            return
        try:
            size, sensitivity = self.size.get(), self.sensitivity.get()
            if not .5 <= sensitivity <= 2:
                raise ValueError("灵敏度须在 0.5 到 2 之间。")
        except (tk.TclError, ValueError) as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.manual_points = None
        self._busy(True)
        self.status.set("正在定位网格、校正透视并识别棋子……")
        self.future = self.executor.submit(recognize, self.image.copy(), size, corners, sensitivity)
        self.root.after(100, self._poll)

    def _poll(self):
        if not self.future.done():
            self.root.after(100, self._poll)
            return
        try:
            self.result = self.future.result()
            self.board = self.result.board.copy()
            self.neutral_points.clear()
            self.history.clear()
            self.refresh()
            self.tabs.select(self.photo_canvas)
            warning = " ".join(self.result.warnings)
            self.status.set(warning or "识别完成。请核对棋盘；左键修改识别，右键将有争议的空区标为中立。")
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showwarning("识别未完成", str(exc))
        finally:
            self.future = None
            self._busy(False)

    def _clear_result(self):
        self.result = self.board = None
        self.history.clear()
        self.neutral_points.clear()
        self.manual_points = None
        for canvas in (self.photo_canvas, self.board_canvas):
            canvas.show(None)
        self.black_text.set("黑方  —")
        self.white_text.set("白方  —")
        self.neutral_text.set("中立  —")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("棋盘图片", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                                                     ("所有文件", "*.*")])
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return
        self._clear_result()
        self.image, self.source = image, Path(path)
        self.original_canvas.show(image)
        self.tabs.select(self.original_canvas)
        self._submit()

    def run_auto(self):
        if self.image is not None:
            self._submit()

    def reclassify(self):
        if self.result is not None and self.image is not None:
            self._submit(self.result.corners)

    def size_changed(self, event=None):
        if self.future:
            return
        self._clear_result()
        if self.image is not None:
            self.original_canvas.show(self.image)
            self._submit()

    def start_manual(self):
        if self.image is None:
            self.status.set("请先打开图片。")
            return
        self.manual_points = []
        self.tabs.select(self.original_canvas)
        self.original_canvas.show(self.image)
        self.status.set("依次点击网格最外侧的：左上 → 右上 → 右下 → 左下。不是木板边缘；Esc 取消。")

    def cancel_manual(self):
        if self.manual_points is not None:
            self.manual_points = None
            self.refresh_source()
            self.status.set("已取消四角标定。")

    def source_click(self, x, y):
        if self.manual_points is None or self.future:
            return
        self.manual_points.append((x, y))
        self.refresh_source()
        if len(self.manual_points) == 4:
            self._submit(self.manual_points)
        else:
            self.status.set(f"已选择 {len(self.manual_points)} / 4 个角点，继续按 左上→右上→右下→左下 选择。")

    def refresh_source(self):
        if self.image is None:
            return
        image = self.image.copy()
        points = self.manual_points if self.manual_points is not None else (self.result.corners if self.result else [])
        radius = max(4, round(max(image.shape[:2]) / 180))
        for i, (x, y) in enumerate(points):
            xy = (round(float(x)), round(float(y)))
            cv2.circle(image, xy, radius, (40, 150, 255), -1)
            cv2.putText(image, str(i + 1), (xy[0] + radius, xy[1] - radius), cv2.FONT_HERSHEY_SIMPLEX,
                         radius / 7, (20, 30, 220), max(1, radius // 3))
        if len(points) == 4:
            cv2.polylines(image, [np.rint(points).astype(np.int32)], True, (70, 190, 70), max(1, radius // 2))
        self.original_canvas.show(image)

    def _intersection(self, x, y):
        if self.board is None or self.future:
            return None
        grid = grid_coordinates(len(self.board))
        col, row = int(np.argmin(abs(grid - x))), int(np.argmin(abs(grid - y)))
        if abs(grid[col] - x) > (grid[1] - grid[0]) * .48 or abs(grid[row] - y) > (grid[1] - grid[0]) * .48:
            return None
        return row, col

    def _remember(self):
        self.history.append((self.board.copy(), self.result.confidence.copy() if self.result else None,
                             self.neutral_points.copy()))
        self.history = self.history[-100:]

    def board_click(self, x, y):
        point = self._intersection(x, y)
        if point is None:
            return
        self._remember()
        old = int(self.board[point])
        self.board[point] = {"黑子": 1, "白子": 2, "空点": 0}.get(self.mode.get(), (old + 1) % 3)
        self.neutral_points.clear()
        if self.result:
            self.result.confidence[point] = 1.
        self.refresh()
        self.status.set("已修改棋子并重新计数；中立标记已清空，可右键重新标记。Ctrl+Z 撤销。")

    def neutral_click(self, x, y):
        point = self._intersection(x, y)
        if point is None or self.board[point] != 0:
            return
        # Toggle all neutral seeds in this connected empty region.
        region, pending = set(), [point]
        n = len(self.board)
        while pending:
            r, c = pending.pop()
            if (r, c) in region:
                continue
            region.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < n and 0 <= nc < n and self.board[nr, nc] == 0 and (nr, nc) not in region:
                    pending.append((nr, nc))
        self._remember()
        if self.neutral_points & region:
            self.neutral_points -= region
        else:
            self.neutral_points.add(point)
        self.refresh()
        self.status.set("已更新整块空区的中立标记。此标记不改变棋子，可再次右键取消。")

    def undo(self):
        if self.future or not self.history:
            return
        self.board, confidence, self.neutral_points = self.history.pop()
        if self.result:
            self.result.confidence = confidence
        self.refresh()
        self.status.set("已撤销上次修改。")

    def refresh(self):
        if self.board is None:
            return
        score = count_area(self.board, self.neutral_points)
        self.black_text.set(f"黑方 {score.black_area}   子 {score.black_stones} + 空 {score.black_territory}")
        self.white_text.set(f"白方 {score.white_area}   子 {score.white_stones} + 空 {score.white_territory}")
        self.neutral_text.set(f"中立 {score.neutral} / {len(self.board) ** 2} 点")
        confidence = self.result.confidence if self.result else None
        self.board_canvas.show(draw_board(self.board, confidence, self.neutral_points))
        if self.result:
            self.photo_canvas.show(draw_board(self.board, confidence, self.neutral_points, self.result.corrected))
        self.refresh_source()

    def export(self):
        if self.board is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="围棋计数结果.json",
                                           filetypes=[("结果 JSON（同时导出 PNG）", "*.json")])
        if not path:
            return
        try:
            paths = export_result(path, self.board, recognition=self.result, source=self.source,
                                  neutral_points=self.neutral_points)
            self.status.set("已保存：" + "、".join(p.name for p in paths))
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def load_result(self):
        path = filedialog.askopenfilename(filetypes=[("结果 JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            board = np.asarray(data["board"])
            if len(board) not in (9, 13, 19):
                raise ValueError("仅支持 9、13、19 路棋盘。")
            points = {tuple(p) for p in data.get("neutral_points", [])}
            count_area(board, points)
            self._clear_result()
            self.image = self.source = None
            self.original_canvas.show(None)
            self.board, self.neutral_points = board.astype(np.uint8), points
            self.size.set(len(board))
            self.refresh()
            self.tabs.select(self.board_canvas)
            self.status.set("已载入棋盘和中立标记，可继续修改并导出。原照片请通过“打开图片”单独加载。")
        except Exception as exc:
            messagebox.showerror("载入失败", str(exc))

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch():
    root = tk.Tk()
    GoCounterApp(root)
    root.mainloop()
