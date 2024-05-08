import cv2

def extract_images(video_path, output_folder):
    # Buka file video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: File video tidak dapat dibuka.")
        return
    
    # Pastikan folder output ada
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    # Hitung jumlah frame dan frame per detik (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)
    
    # Mulai proses ekstraksi gambar
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Hitung waktu (detik) dari frame saat ini
        time_in_seconds = count / fps
        
        # Jika waktu saat ini adalah kelipatan detik, simpan gambar
        if int(time_in_seconds) != int((count - 1) / fps):
            image_path = os.path.join(output_folder, f"frame_{int(time_in_seconds)}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Frame {count} (Detik {int(time_in_seconds)}) disimpan.")
        
        count += 1
    
    cap.release()
    print("Selesai.")

# Path file video
video_path = r"D:\TUGAS KULIAH\PORTO\Traffic Counting YOLOv8\traffics.mp4"
# Folder untuk menyimpan gambar
output_folder = r"D:\TUGAS KULIAH\PORTO\Traffic Counting YOLOv8\Image capture"

# Ekstrak gambar dengan satu gambar setiap detik
extract_images(video_path, output_folder)
