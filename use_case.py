#initiate banyak kendaraan
#timur = 10
#selatan = 5
#barat = 8
# utara = 2
import RPi.GPIO as GPIO
import time
import datetime
import tm1637
from Class.Class import Traffic
import asyncio
kendaraan_timur_pertama = 10
kendaraan_selatan_pertama = 5
kendaraan_barat_pertama = 8
kendaraan_utara_pertama = 2

timur = Traffic("timur",5,[22,27,17])
selatan = Traffic("selatan",6,[21,20,16]) 
barat = Traffic("barat",13,[7,8,25])
utara = Traffic("utara",19,[18,15,14])

timur.setGreenTime(timur.countGreenTime(kendaraan_timur_pertama))
selatan.setGreenTime(selatan.countGreenTime(kendaraan_selatan_pertama))
barat.setGreenTime(selatan.countGreenTime(kendaraan_barat_pertama))
utara.setGreenTime(utara.countGreenTime(kendaraan_utara_pertama))

timur.setRedTime(0)
selatan.setRedTime(selatan.countRedTime(timur))
barat.setRedTime(barat.countRedTime(selatan))
utara.setRedTime(timur.countRedTime(barat))

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

tm_timur= tm1637.TM1637(clk=timur.getPinTraffic()[0],dio=timur.getDio())
tm_selatan = tm1637.TM1637(clk=selatan.getPinTraffic()[0],dio=selatan.getDio())
tm_barat = tm1637.TM1637(clk=barat.getPinTraffic()[0],dio=barat.getDio())
tm_utara = tm1637.TM1637(clk=utara.getPinTraffic()[0],dio=utara.getDio())

green = "green"
red = "red"
yellow = "yellow"

try:
    while(True):    
        timur.light_on(green)
        selatan.light_on(red)
        barat.light_on(red)
        utara.light_on(red)

        for i in range(0,utara.getRedTime()):
            if timur.getGreenTime()==1:
                timur.light_on(yellow)
                time.sleep(1)
                timur.updateTime(green)
                timur.light_on(red)
                tm_timur.numbers(00,timur.getRedTime())
                selatan.light_on(yellow)
                time.sleep(1)
                selatan.light_on(green)
                tm_selatan.numbers(00,selatan.getGreenTime())
            tm_timur.numbers(00,timur.getGreenTime())
            tm_selatan.numbers(00,selatan.getRedTime())
            tm_barat.numbers(00,barat.getRedTime())
            tm_utara.numbers(00,utara.getRedTime())
            time.sleep(1)
            timur.updateTime(green)
            selatan.updateTime(red)
            barat.updateTime(red)
            utara.updateTime(red)
            i+=1

except KeyboardInterrupt:
    tm_timur.write([0, 0, 0, 0])
    tm_selatan.write([0,0,0,0])
    tm_barat.write([0,0,0,0])
    tm_utara.write([0,0,0,0])    
    GPIO.cleanup()
