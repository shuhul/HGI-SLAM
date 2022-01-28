import subprocess
import threading
import time

def run():
    for i in range(5):
        wait(0.3)
        print("From thread!")

def wait(secs):
    start = time.time()
    while(time.time() < start + secs):
        pass

if __name__ == "__main__":
    thread = threading.Thread(target=run)
    thread.start()
    subprocess.call(['sh', './orbslam.sh'])
    thread.join()
    print("End of process")