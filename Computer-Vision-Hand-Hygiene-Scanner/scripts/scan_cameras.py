import cv2
import sys

def find_cameras(max_index=10):
    available = []
    backends = [cv2.CAP_ANY]
    # On Windows, try DirectShow backend which often works with external webcams
    if sys.platform.startswith("win"):
        backends.insert(0, cv2.CAP_DSHOW)
    
    for i in range(max_index + 1):
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(i, backend)
            if cap is None:
                continue
            is_opened = cap.isOpened()
            ok = False
            if is_opened:
                ok, _ = cap.read()
            if is_opened and ok:
                available.append(i)
                cap.release()
                break
            if cap is not None:
                cap.release()
    return available

if __name__ == '__main__':
    cams = find_cameras(10)
    print('Available camera indices:', cams)
