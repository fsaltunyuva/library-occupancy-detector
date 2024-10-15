import socket
# Sunucu bilgileri
HOST = '127.0.0.1'  # Localhost
PORT = 65432  # Bağlanılacak port numarası

# TCP/IP socket oluştur
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))  # Sunucuyu belirtilen IP ve port üzerinde çalıştır
    s.listen()  # Gelen bağlantıları dinle
    print(f"Sunucu {HOST}:{PORT} üzerinde dinleniyor...")

    # Bir istemcinin bağlanmasını bekle
    conn, addr = s.accept()
    with conn:
        print(f"Bağlantı kuruldu: {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"İstemciden gelen veri: {data.decode()}")
            conn.sendall(data)  # Gelen veriyi geri gönder (Echo Server)
