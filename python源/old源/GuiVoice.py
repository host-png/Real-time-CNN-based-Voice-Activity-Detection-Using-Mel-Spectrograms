import sys
import os
import time
import ctypes
import threading
import sounddevice as sd
import soundfile as sf

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal

# ================== 配置 ==================
OUTPUT_DIR = "recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== DLL 录音线程 ==================
class AudioRecorderDLL(QThread):
    finished = Signal(str)

    def __init__(self, category="voice"):
        super().__init__()
        self.running = False
        self.category = category

        #Dll
        self.dll = ctypes.CDLL(r"E:\AI\PythonProject1\Dll1.dll")

        #DLL参数
        self.dll.start_capture.argtypes = [ctypes.c_char_p]
        self.dll.start_capture.restype = None
        self.dll.stop_capture.restype = None
        self.dll.get_total_samples.restype = ctypes.c_int

    def run(self):
        self.running = True

        # 根据类别生成文件名
        prefix = {
            "人声": "voice",
            "非人声": "no_voice",
            "NVoice": "nvoice"
        }.get(self.category, "voice")

        filename = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        self.filepath = os.path.join(OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # 开始 DLL 录音
        self.dll.start_capture(self.filepath.encode("utf-8"))

        # 等待停止信号
        while self.running:
            self.msleep(100)

        # 停止录音
        self.dll.stop_capture()
        self.finished.emit(self.filepath)

    def stop(self):
        self.running = False

# ================== 悬浮窗 ==================
class FloatingWindow(QWidget):
    run_in_main = Signal(object)  # 用于线程安全调用 GUI

    def __init__(self):
        super().__init__()

        # ---- 窗口属性 ----
        self.setWindowTitle("数据采集器 DLL")
        self.setFixedSize(220, 180)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setWindowOpacity(0.92)

        # ---- 状态 ----
        self.recorder = None
        self.last_file = None
        self._drag_pos = None
        self.is_collapsed = False
        self.normal_height = 180

        # ---- 顶部栏 ----
        self.title = QLabel(" 数据采集器 DLL")
        self.btn_min = QPushButton("—")
        self.btn_close = QPushButton("×")
        self.btn_min.setFixedWidth(28)
        self.btn_close.setFixedWidth(28)
        self.btn_min.clicked.connect(self.toggle_minimize)
        self.btn_close.clicked.connect(self.close)

        top = QHBoxLayout()
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.btn_min)
        top.addWidget(self.btn_close)

        # ---- 类别选择下拉框 ----
        self.category_select = QComboBox()
        self.category_select.addItems(["人声", "非人声", "NVoice"])
        self.category_select.setCurrentIndex(0)

        # ---- 功能按钮 ----
        self.btn_record = QPushButton("▶ 开始录制")
        self.btn_check = QPushButton(" 检查录音")
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_check.clicked.connect(self.check_audio)

        # ---- 主布局 ----
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(QLabel("录音类别:"))
        layout.addWidget(self.category_select)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.btn_check)
        self.setLayout(layout)

        # ---- 信号绑定 ----
        self.run_in_main.connect(lambda func: func())

    # ---------- 拖动窗口 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    # ---------- 缩小 / 展开 ----------
    def toggle_minimize(self):
        if not self.is_collapsed:
            self.setFixedHeight(32)
            self.btn_record.hide()
            self.btn_check.hide()
            self.category_select.hide()
            self.is_collapsed = True
        else:
            self.setFixedHeight(self.normal_height)
            self.btn_record.show()
            self.btn_check.show()
            self.category_select.show()
            self.is_collapsed = False

    # ---------- 录制 ----------
    def toggle_record(self):
        if self.recorder and self.recorder.isRunning():
            self.btn_record.setText("▶ 开始录制")
            self.recorder.stop()
        else:
            category = self.category_select.currentText()
            self.recorder = AudioRecorderDLL(category=category)
            self.recorder.finished.connect(self.record_finished)
            self.recorder.start()
            self.btn_record.setText("■ 停止录制")

    def record_finished(self, path):
        self.last_file = path
        QMessageBox.information(self, "完成", f"录音已保存：\n{path}")

    # ---------- 检查 ----------
    def check_audio(self):
        if not self.last_file or not os.path.exists(self.last_file):
            QMessageBox.warning(self, "提示", "没有可检查的录音")
            return

        # 播放音频线程
        def play_thread():
            data, sr = sf.read(self.last_file, dtype="float32")
            sd.play(data, sr)
            sd.wait()

            # 弹窗删除确认在主线程执行
            def confirm_delete():
                reply = QMessageBox.question(
                    self,
                    "是否删除？",
                    "是否删除这条录音？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    os.remove(self.last_file)
                    self.last_file = None
                    QMessageBox.information(self, "已删除", "录音已删除")

            self.run_in_main.emit(confirm_delete)

        threading.Thread(target=play_thread, daemon=True).start()

    # ---------- 安全关闭 ----------
    def closeEvent(self, event):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait()
        event.accept()


# ================== main ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FloatingWindow()
    win.show()
    sys.exit(app.exec())
