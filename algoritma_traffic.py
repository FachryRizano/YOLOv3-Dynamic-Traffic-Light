import time
kendaraan_timur_pertama = 10
kendaraan_selatan_pertama = 5
kendaraan_barat_pertama = 8
kendaraan_utara_pertama = 2



def total_waktu_hijau(ruas_jalan):
    threshold = 15
    waktu = ruas_jalan*2
    if waktu >= threshold:
        return threshold
    else:
        return waktu


waktu_hijau_timur = total_waktu_hijau(kendaraan_timur_pertama)
waktu_hijau_selatan = total_waktu_hijau(kendaraan_selatan_pertama)
waktu_hijau_barat = total_waktu_hijau(kendaraan_barat_pertama)
waktu_hijau_utara = total_waktu_hijau(kendaraan_utara_pertama)

waktu_ruas={
    'timur':[waktu_hijau_timur,0],
    'selatan':[waktu_hijau_selatan,0],
    'barat':[waktu_hijau_barat,0],
    'utara':[waktu_hijau_utara,0]
}


def total_waktu_merah(waktu_ruas):
    waktu_ruas['selatan'][1] = waktu_ruas['timur'][0] + waktu_ruas['timur'][1] + 3 
    waktu_ruas['barat'][1] = waktu_ruas['selatan'][0] + waktu_ruas['selatan'][1] + 3 
    waktu_ruas['utara'][1] = waktu_ruas['barat'][0] + waktu_ruas['barat'][1] + 3
    return waktu_ruas

def countdown(waktu_ruas):
    print("==== WAKTU LAMPU ====")

    while waktu_ruas['timur'][0] != -1 : 
        print("T S B U")
        print(waktu_ruas['timur'][0],waktu_ruas['selatan'][1],waktu_ruas['barat'][1],waktu_ruas['utara'][1])
        time.sleep(1)
        waktu_ruas['timur'][0] -=1     
        waktu_ruas['selatan'][1] -=1
        waktu_ruas['barat'][1] -=1
        waktu_ruas['utara'][1] -=1

waktu_ruas = total_waktu_merah(waktu_ruas)

# ketika timur hijau, yang lainnya merah
countdown(waktu_ruas)
#pinndah ke jalur selanjutnya

