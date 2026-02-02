import ctypes

dll = ctypes.CDLL(r"E:\AI\PythonProject1\Dll1.dll"
                  r"")  # DLL 路径

dll.start_capture.argtypes = [ctypes.c_char_p]
dll.start_capture.restype = None
dll.stop_capture.restype = None
dll.get_total_samples.restype = ctypes.c_int

filename = b"E:\\AI\\PythonProject1\\recordings\\test_capture.wav"
dll.start_capture(filename)

input("录音中，按回车停止...")

dll.stop_capture()
n_samples = dll.get_total_samples()
print("录音完成，16k样本数:", n_samples)
