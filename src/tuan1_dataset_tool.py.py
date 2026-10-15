import cv2
import os
import numpy as np

def tao_thu_muc():
    folders = ['dataset/raw', 'dataset/real', 'dataset/tampered']
    for f in folders:
        os.makedirs(f, exist_ok=True)
    print("tao xong folder")

def che_thong_tin(img_name):
    path = os.path.join('dataset/raw', img_name)
    img = cv2.imread(path)

    if img is None:
        print(f"loi ko thay anh {path}")
        return

    clone = img.copy()
    
    win_name = "Tool che PII"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 1000)

    while True:
        # chon vung roi nhan space, thoat nhan esc
        roi = cv2.selectROI(win_name, clone, fromCenter=False, showCrosshair=True)
        
        if roi == (0, 0, 0, 0):
            break
            
        x, y, w, h = roi
        roi_img = clone[y:y+h, x:x+w]
        
        # blur manh
        blur_roi = cv2.GaussianBlur(roi_img, (51, 51), 0)
        clone[y:y+h, x:x+w] = blur_roi
        
        cv2.imshow(win_name, clone)

    save_path = os.path.join('dataset/real', "anonymized_" + img_name)
    cv2.imwrite(save_path, clone)
    print("luu xong tai:", save_path)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tao_thu_muc()
    
    # file test
    test_file = 'sodo1.jpg' 
    # che_thong_tin(test_file)