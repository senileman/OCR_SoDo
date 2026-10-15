# pip install scikit-learn jiwer seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import jiwer

def evaluate_ocr(ground_truth_list, hypothesis_list):
    gt_clean = [gt.strip().lower() for gt in ground_truth_list]
    hyp_clean = [hyp.strip().lower() for hyp in hypothesis_list]
    
    wer_score = jiwer.wer(gt_clean, hyp_clean)
    
    cer_score = jiwer.cer(gt_clean, hyp_clean)
    
    print("--- KẾT QUẢ ĐÁNH GIÁ OCR ---")
    print(f"Word Error Rate (WER): {wer_score:.4f} (Tức {wer_score*100:.2f}%)")
    print(f"Character Error Rate (CER): {cer_score:.4f} (Tức {cer_score*100:.2f}%)")
    
    return wer_score, cer_score

def evaluate_classification(y_true, y_pred, y_scores=None):
    """
    Đánh giá mô hình phân loại (anh thật = 0, anhr giả/cắt ghép = 1).
    - y_true: Nhãn thực tế (ví dụ: [0, 1, 1, 0, 1])
    - y_pred: Nhãn mô hình dự đoán (ví dụ: [0, 1, 0, 0, 1])
    - y_scores: Điểm xác suất để vẽ ROC
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ PHÁT HIỆN GIAN LẬN ---")
    print(f"Accuracy (Độ chính xác tổng thể): {acc:.4f}")
    print(f"Precision (Độ chuẩn xác): {prec:.4f}")
    print(f"Recall (Độ bao phủ): {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Thật (Real)', 'Giả (Tampered)'], 
                yticklabels=['Thật (Real)', 'Giả (Tampered)'])
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
    plt.ylabel('Nhãn Thực Tế')
    plt.xlabel('Nhãn Dự Đoán')
    plt.savefig("../output/confusion_matrix.png")
    plt.close()
    
    if y_scores is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig("../output/roc_curve.png")
        plt.close()
        print(f"Đã lưu biểu đồ ROC và Confusion Matrix vào thư mục output!")

def evaluate_geometry_error(true_lengths, pred_lengths):
    """
    Tính sai số tuyệt đối trung bình (MAE) của việc đo đạc cạnh thửa đất.
    """
    if len(true_lengths) != len(pred_lengths):
        print("Số lượng cạnh thực tế và dự đoán không khớp!")
        return None
        
    true_arr = np.array(true_lengths)
    pred_arr = np.array(pred_lengths)
    
    errors = np.abs(true_arr - pred_arr)
    mae = np.mean(errors)
    
    print("\n--- KẾT QUẢ TRÍCH XUẤT HÌNH HỌC ---")
    print(f"Sai số tuyệt đối trung bình (MAE): {mae:.2f} pixels")
    return mae

if __name__ == "__main__":
    gt_texts = ["Bà Nguyễn Thị A", "Thửa đất số: 123", "Diện tích: 100 m2"]
    hyp_texts = ["Bà Nguyễn Thị A", "Thửa dất sô: 123", "Diện tích: 100 m2"] 
    evaluate_ocr(gt_texts, hyp_texts)
    
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1]
    y_scores = [0.1, 0.6, 0.2, 0.3, 0.4, 0.8, 0.9, 0.4, 0.7, 0.85]
    
    evaluate_classification(y_true, y_pred, y_scores)
    
    true_len = [150.5, 200.0, 150.5, 200.0]
    pred_len = [148.0, 202.5, 149.0, 198.5]
    evaluate_geometry_error(true_len, pred_len)