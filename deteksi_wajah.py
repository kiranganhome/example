import cv2
import sys

# Nama file gambar yang akan diproses.
# Ganti dengan nama file gambarmu, contoh: 'keluarga.jpg'
image_path = 'foto_orang.jpg'

# Nama file XML untuk model deteksi wajah
cascade_path = 'haarcascade_frontalface_default.xml'

# Periksa apakah file gambar dan cascade ada
try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise FileNotFoundError(f"File cascade tidak ditemukan: {cascade_path}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit()

# Mengubah gambar menjadi skala abu-abu (grayscale) untuk pemrosesan yang lebih cepat
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Deteksi wajah dalam gambar
# Parameter:
# - gray: gambar skala abu-abu
# - scaleFactor: seberapa banyak ukuran gambar dikurangi di setiap skala gambar
# - minNeighbors: seberapa banyak tetangga yang harus dimiliki setiap kandidat persegi panjang
# - minSize: ukuran minimum objek yang akan dideteksi
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

# Menggambar kotak di sekeliling setiap wajah yang terdeteksi
for (x, y, w, h) in faces:
    # Menggambar persegi panjang di gambar
    # Parameter:
    # - image: gambar tempat persegi panjang digambar
    # - (x, y): koordinat sudut kiri atas
    # - (x+w, y+h): koordinat sudut kanan bawah
    # - (255, 0, 0): warna kotak dalam format BGR (biru)
    # - 2: ketebalan garis
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Menampilkan jumlah wajah yang terdeteksi di konsol
print(f"Ditemukan {len(faces)} wajah di dalam gambar.")

# Menampilkan gambar dengan kotak deteksi wajah
cv2.imshow('Deteksi Wajah dengan OpenCV', image)

# Menunggu pengguna menekan tombol apa pun sebelum menutup jendela
cv2.waitKey(0)

# Menutup semua jendela OpenCV
cv2.destroyAllWindows()
