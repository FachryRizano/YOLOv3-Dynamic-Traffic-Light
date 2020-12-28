#initiate banyak kendaraan
#timur = 10
#selatan = 5
#barat = 8
# utara = 2
import RPi.GPIO as GPIO
import time
import datetime
import tm1637

kendaraan_timur_pertama = 10
kendaraan_selatan_pertama = 5
kendaraan_barat_pertama = 8
kendaraan_utara_pertama = 2

#penghitungan total waktu
# jumlah kendaraan * 2  = waktu lampu hijau
 # threshold waktu = 15
ruas={
    'timur':[22,27,17],
    'selatan':[21,20,16],
    'barat':[7,8,25],
    'utara':[18,15,14]
    }
#index 0 = total waktu hijau
#index 1 =total waktu merah


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
    # for i in range(0,3):   
    # current_ruas = list(waktu_ruas.values())[i]
    # merah_ruas_next = current_ruas[0] + current_ruas[1] + 3

    waktu_ruas['selatan'][1] = waktu_ruas['timur'][0] + waktu_ruas['timur'][1] + 3 
    waktu_ruas['barat'][1] = waktu_ruas['selatan'][0] + waktu_ruas['selatan'][1] + 3 
    waktu_ruas['utara'][1] = waktu_ruas['barat'][0] + waktu_ruas['barat'][1] + 3


    # print(waktu_ruas)
    return waktu_ruas

        # waktu_ruas[current_ruas][1].values() = mereah_ruas_next
    # print(waktu_ruas)

print(total_waktu_merah(waktu_ruas))  


GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM)

for arah in ruas.values():
    for i in range(3):
        GPIO.setup(arah[i],GPIO.OUT)
        GPIO.output(arah[i],False)
    # GPIO.setup(arah[1],GPIO.OUT)
    # GPIO.setup(arah[2],GPIO.OUT)


tm = tm1637.TM1637(clk=ruas['timur'][0],dio=5)
tm_2 = tm1637.TM1637(clk=ruas['selatan'][0],dio=6)
tm_3 = tm1637.TM1637(clk=ruas['barat'][0],dio=13)
tm_4 = tm1637.TM1637(clk=ruas['utara'][0],dio=19)

#initially turn 

try:

    while(True):
        total_waktu_merah(waktu_ruas)

        tm.numbers(00, waktu_ruas['timur'][0])
        tm_2.numbers(00, waktu_ruas['selatan'][1])
        tm_3.numbers(00, waktu_ruas['barat'][1])
        tm_4.numbers(00, waktu_ruas['utara'][1])

        GPIO.output(ruas['timur'][0],True)
        GPIO.output(ruas['timur'][1],False)
        GPIO.output(ruas['timur'][2],False)
        
        GPIO.output(ruas['selatan'][0],False)
        GPIO.output(ruas['selatan'][1],False)
        GPIO.output(ruas['selatan'][2],True)

        GPIO.output(ruas['barat'][0],False)
        GPIO.output(ruas['barat'][1],False)
        GPIO.output(ruas['barat'][2],True)

        GPIO.output(ruas['utara'][0],False)
        GPIO.output(ruas['utara'][1],False)
        GPIO.output(ruas['utara'][2],True)

        # time.sleep(0.1)
        # GPIO.output(ruas['timur'][0],False) 
        # GPIO.output(ruas['timur'][0],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['timur'][0],False)
        # GPIO.output(ruas['timur'][1],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['timur'][1],False)
        # GPIO.output(ruas['timur'][2],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['timur'][2],False)
        
        # time.sleep(0.1)
        # GPIO.output(ruas['selatan'][0],False)
        # GPIO.output(ruas['selatan'][0],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['selatan'][0],False)
        # GPIO.output(ruas['selatan'][1],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['selatan'][1],False)
        # GPIO.output(ruas['selatan'][2],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['selatan'][2],False)
        
        # time.sleep(0.1)
        # GPIO.output(ruas['barat'][0],False)
        # GPIO.output(ruas['barat'][0],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['barat'][0],False)
        # GPIO.output(ruas['barat'][1],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['barat'][1],False)
        # GPIO.output(ruas['barat'][2],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['barat'][2],False)
        
        # time.sleep(0.1)
        # GPIO.output(ruas['utara'][0],False)
        # GPIO.output(ruas['utara'][0],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['utara'][0],False)
        # GPIO.output(ruas['utara'][1],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['utara'][1],False)
        # GPIO.output(ruas['utara'][2],True)
        # time.sleep(0.1)
        # GPIO.output(ruas['utara'][2],False)
        
        

except KeyboardInterrupt:
    tm.write([0, 0, 0, 0])
    tm_2.write([0,0,0,0])
    tm_3.write([0,0,0,0])
    tm_4.write([0,0,0,0])    
    GPIO.cleanup()
