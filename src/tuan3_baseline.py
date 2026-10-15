import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pytesseract

# cai tesseract.exe
# doi link folder chua tesseract.exe tuong ung
# unblock file vie.traineddata
pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract/tesseract.exe'

os.environ['TESSDATA_PREFIX'] = 'D:/Tesseract/tessdata'

def extract_red_seal(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # bat mau do (do mau do trong hsv bi chia lam 2 khoang o 2 dau)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2
    
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    seal = cv2.bitwise_and(img, img, mask=red_mask)
    return seal, red_mask

def analyze_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return mag_spectrum

def run_tesseract(img):
    # chuyen sang grayscale va dung otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cfg = r'--oem 1 --psm 6'
    txt = pytesseract.image_to_string(thresh, lang='vie', config=cfg)
    return txt

if __name__ == "__main__":
    out_dir = "dataset/tuan3_results"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/seals", exist_ok=True)
    os.makedirs(f"{out_dir}/fft", exist_ok=True)

    print("1. Dang boc tach dau do o tap Real...")
    seal_paths = glob.glob("dataset/processed_real/seal/*.jpg")
    for p in seal_paths:
        name = os.path.basename(p)
        im = cv2.imread(p)
        if im is not None:
            seal_ex, _ = extract_red_seal(im)
            cv2.imwrite(f"{out_dir}/seals/red_{name}", seal_ex)

    print("2. Dang chay FFT so sanh anh that va anh gia...")
    real_paths = glob.glob("dataset/processed_real/full_clahe/*.jpg")[:5]
    fake_paths = glob.glob("dataset/processed_tampered/full_clahe/*.jpg")[:5]
    
    for rp, fp in zip(real_paths, fake_paths):
        name = os.path.basename(rp)
        im_real = cv2.imread(rp)
        im_fake = cv2.imread(fp)
        
        # crop 1 vung 256x256 giua anh de phan tich pho
        h, w = im_real.shape[:2]
        roi_r = im_real[h//2:h//2+256, w//2:w//2+256]
        roi_f = im_fake[h//2:h//2+256, w//2:w//2+256]
        
        fft_real = analyze_fft(roi_r)
        fft_fake = analyze_fft(roi_f)
        
        # luu 2 pho canh nhau
        plt.figure(figsize=(10,5))
        plt.subplot(121), plt.imshow(fft_real, cmap='gray'), plt.title('FFT - So That')
        plt.subplot(122), plt.imshow(fft_fake, cmap='gray'), plt.title('FFT - Gian lan')
        plt.savefig(f"{out_dir}/fft/compare_{name}")
        plt.close()

    print("3. Dang doc thu OCR 1 anh text...")
    text_paths = glob.glob("dataset/processed_real/text/*.jpg")
    if text_paths:
        test_img = cv2.imread(text_paths[0])
        kq = run_tesseract(test_img)
        print("ket qua text:", kq)
    