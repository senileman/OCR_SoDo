import cv2
import numpy as np
import math
import os
import glob

def enhance_contrast_clahe(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()[:4]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 <= angle <= 45:
                angles.append(angle)
    
    median_angle = np.median(angles) if len(angles) > 0 else 0
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def process_folder(input_folder, output_base):
    for roi in ['text', 'diagram', 'seal', 'full_clahe']:
        os.makedirs(os.path.join(output_base, roi), exist_ok=True)
        
    img_paths = glob.glob(input_folder + "/*.*")
    
    STD_WIDTH = 1200
    STD_HEIGHT = 800
    
    for path in img_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None: continue
            
        deskewed = deskew_image(img)
        standardized_img = cv2.resize(deskewed, (STD_WIDTH, STD_HEIGHT))
        gray_enhanced = enhance_contrast_clahe(standardized_img)
        
        h, w = gray_enhanced.shape[:2]
        text_roi = gray_enhanced[0:int(h*0.55), 0:int(w*0.45)]
        diagram_roi = gray_enhanced[int(h*0.1):int(h*0.8), int(w*0.5):w]
        
        # cat con dau tu mau anh goc da resize
        seal_roi = standardized_img[int(h*0.65):h, 0:int(w*0.5)]
        
        cv2.imwrite(os.path.join(output_base, 'full_clahe', filename), gray_enhanced)
        cv2.imwrite(os.path.join(output_base, 'text', filename), text_roi)
        cv2.imwrite(os.path.join(output_base, 'diagram', filename), diagram_roi)
        cv2.imwrite(os.path.join(output_base, 'seal', filename), seal_roi)
        
        print(f"Xử lý chuẩn xác: {filename}")

if __name__ == "__main__":
    print("--- Dang chay tap Real ---")
    process_folder("dataset/real", "dataset/processed_real")
    print("--- Dang chay tap Tampered ---")
    process_folder("dataset/tampered", "dataset/processed_tampered")
    print("Hoan thanh batch processing!")