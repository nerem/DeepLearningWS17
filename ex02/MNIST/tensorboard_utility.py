import os
import shutil
from time import sleep
import threading
import webbrowser

python_path = "C:\\ProgramData\\Anaconda3\\envs\\tensorflow\\python.exe"
tensorboard_path = "C:\\ProgramData\\Anaconda3\\envs\\tensorflow\\Lib\\site-packages\\tensorboard\\main.py"

def launch_tb(sub_dir = None):
    if sub_dir is None:
        log_dir = os.getcwd() + "\\logs"
    else:
        log_dir = os.getcwd() + "\\logs" + "\\" + sub_dir
    if not os.path.isdir(log_dir):
        print("No logs.")
        return
    def tb():
        os.system("{} {} --logdir={}".format(python_path, tensorboard_path, log_dir))
        return
    t = threading.Thread(target = tb, args=([]))
    t.daemon = True
    t.start()
    sleep(0.5)
    webbrowser.open("http:\\localhost:6006")
    return

def clear_logs():
    shutil.rmtree(os.getcwd() + "\\logs")
    return

def get_log_dir(N = 100, get_latest = False):
    for i in range(1, N + 1):
        log_dir = os.getcwd() + "\\logs\\log{}".format(i)
        if not os.path.isdir(log_dir):
            if get_latest:
                return os.getcwd() + "\\logs\\log{}".format(max(i - 1, 1))
            else:
                return log_dir
    return os.getcwd() + "\\logs\\_log"