import os
import easyocr
import pandas as pd

# 1. KHỞI TẠO
reader = easyocr.Reader(['en']) # Đọc tiếng Anh/Số
base_path = os.path.dirname(os.path.abspath(__file__))
crop_dir = os.path.join(base_path, "extracted_all")
results = []

print("--- ĐANG BẮT ĐẦU ĐỌC CHỮ TỪ CÁC MẢNH CẮT ---")

# 2. QUÉT QUA CÁC THƯ MỤC ẢNH ĐÃ CẮT
for folder_name in os.listdir(crop_dir):
    folder_path = os.path.join(crop_dir, folder_name)
    if not os.path.isdir(folder_path): continue

    print(f"Đang xử lý bản vẽ: {folder_name}...")
    
    for img_name in os.listdir(folder_path):
        # Chỉ đọc chữ từ các mảnh Note hoặc Table (PartDrawing thường là hình, không cần đọc)
        if "Note" in img_name or "Table" in img_name:
            img_path = os.path.join(folder_path, img_name)
            
            # Đọc chữ bằng OCR
            ocr_result = reader.readtext(img_path, detail=0)
            full_text = " ".join(ocr_result)
            
            # Lưu vào danh sách
            results.append({
                "Bản vẽ gốc": folder_name,
                "Loại đối tượng": "Note" if "Note" in img_name else "Table",
                "Nội dung trích xuất": full_text
            })

# 3. XUẤT RA FILE EXCEL
df = pd.DataFrame(results)
output_excel = os.path.join(base_path, "ket_qua_cuoi_cung.xlsx")
df.to_excel(output_excel, index=False)

print(f"\n--- THÀNH CÔNG RỰC RỠ! ---")
print(f"Toàn bộ dữ liệu đã được lưu vào file: {output_excel}")