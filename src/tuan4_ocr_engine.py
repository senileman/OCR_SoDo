# pip install scikit-image paddlepaddle paddleocr

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from paddleocr import PaddleOCR
import math

# use_angle_cls=True để tự động xoay chữ bị nghiêng nhẹ
ocr_model = PaddleOCR(use_angle_cls=True, lang='vi')

def extract_diagram_geometry(diagram_roi):
    gray = cv2.cvtColor(diagram_roi, cv2.COLOR_BGR2GRAY)
    
    # Phân ngưỡng nghịch đảo để nét vẽ thành màu trắng trên nền đen
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Rút trích khung xương bằng Morphological
    skeleton = np.zeros(thresh.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    
    while not done:
        eroded = cv2.erode(thresh, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(thresh, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        thresh = eroded.copy()
        
        if cv2.countNonZero(thresh) == 0:
            done = True
            
    # Dùng HoughLinesP để tìm các đoạn thẳng trên khung xương
    lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    
    result_img = diagram_roi.copy()
    edge_lengths = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            edge_lengths.append(length)
            
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    return result_img, edge_lengths

def advanced_ocr_pipeline(text_roi):
    gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 3))
    
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted_data = []
    
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:
            line_img = text_roi[y:y+h, x:x+w]
            
            result = ocr_model.ocr(line_img, cls=True)
            if result and result[0]:
                for res in result[0]:
                    text = res[1][0]
                    confidence = res[1][1]
                    extracted_data.append((text, confidence))
                    
    return extracted_data

def verify_seal_lbp(seal_img, reference_seal_img):

    def get_lbp_hist(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Cài đặt tham số LBP
        radius = 1
        n_points = 8 * radius
        
        # Tính toán ma trận LBP
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        
        # Tạo Histogram của LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist

    hist1 = get_lbp_hist(seal_img)
    hist2 = get_lbp_hist(reference_seal_img)
    
    hist1 = np.float32(hist1)
    hist2 = np.float32(hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    return score

if __name__ == "__main__":
    print("OK")