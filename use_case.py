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



timur.setRedTime(3)
selatan.setRedTime(selatan.countRedTime(timur))
barat.setRedTime(barat.countRedTime(selatan))
utara.setRedTime(utara.countRedTime(barat))
# selatan.setRedTime(10)
# barat.setRedTime(10)
# utara.setRedTime(10)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

tm_timur= tm1637.TM1637(clk=timur.getPinTraffic()[0],dio=timur.getDio())
tm_selatan = tm1637.TM1637(clk=selatan.getPinTraffic()[0],dio=selatan.getDio())
tm_barat = tm1637.TM1637(clk=barat.getPinTraffic()[0],dio=barat.getDio())
tm_utara = tm1637.TM1637(clk=utara.getPinTraffic()[0],dio=utara.getDio())



green = "green"
red = "red"
yellow = "yellow"
async def timurCountdown():
    while timur.getGreenTime() > -1:
        tm_timur.numbers(00,timur.getGreenTime())
        tm_selatan.numbers(00,selatan.getRedTime())
        tm_barat.numbers(00,barat.getRedTime())
        tm_utara.numbers(00,utara.getRedTime())
        await asyncio.sleep(1)
        timur.updateTime(green)
        selatan.updateTime(red)
        barat.updateTime(red)
        utara.updateTime(red)
        if timur.getGreenTime() == 0:
            break


async def selatanCountdown():
    while selatan.getGreenTime() > -1:
        tm_timur.numbers(00,timur.getRedTime())
        tm_selatan.numbers(00,selatan.getGreenTime())
        tm_barat.numbers(00,barat.getRedTime())
        tm_utara.numbers(00,utara.getRedTime())
        await asyncio.sleep(1)
        timur.updateTime(red)
        selatan.updateTime(green)
        barat.updateTime(red)
        utara.updateTime(red)    
    
async def baratCountdown():
    while barat.getGreenTime() > -1:
        tm_timur.numbers(00,timur.getRedTime())
        tm_selatan.numbers(00,selatan.getRedTime())
        tm_barat.numbers(00,barat.getGreenTime())
        tm_utara.numbers(00,utara.getRedTime())
        await asyncio.sleep(1)
        timur.updateTime(red)
        selatan.updateTime(red)
        barat.updateTime(green)
        utara.updateTime(red)
    utaraCountdown()

async def utaraCountdown():
    while utara.getGreenTime() > -1:
        tm_timur.numbers(00,timur.getRedTime())
        tm_selatan.numbers(00,selatan.getRedTime())
        tm_barat.numbers(00,barat.getRedTime())
        tm_utara.numbers(00,utara.getGreenTime())
        await asyncio.sleep(1)
        timur.updateTime(red)
        selatan.updateTime(red)
        barat.updateTime(red)
        utara.updateTime(green)

async def transisiLampu():
    timur.light_on(yellow)
    await asyncio.sleep(1)
    
    timur.light_on(red)
    await asyncio.sleep(1)
    selatan.light_on(yellow)
    await asyncio.sleep(1)
    selatan.light_on(green)
    await asyncio.sleep(selatan.getGreenTime())

#Fungsi transisi lampu
# async def lampTransition(object):
#     if obj.getGreenTime() == 0:
#         obj.light_on(yellow)
#         await asyncio.sleep(1)
#         obj.light_on(red)
#     else obj.getRedTime()==0:
#         obj.light_on(yellow)
#         await asyncio.sleep(1)
#         obj.light_on(green)
#Fungsi transisi waktu
# async def timeTransition():
    #set waktu ketika lampu merah nyala
    #set waktu ketika lampu hijau nyala

#Fungsi transisi lampu 
#jika durasi greentime abis maka ganti lampu jadi kuning sedetik, setelah itu merah

async def main():
    timur.light_on(green)
    selatan.light_on(red)
    barat.light_on(red)
    utara.light_on(red)
    asyncio.gather(timurCountdown())
    await asyncio.sleep(timur.getGreenTime())
    asyncio.gather(transisiLampu())
    asyncio.gather(selatanCountdown())
    await asyncio.sleep(selatan.getGreenTime())
    
    # await asyncio.sleep(3)
    # loop.stop()

if __name__ == "__main__":
    try:    
        while(True):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
    except KeyboardInterrupt:
        tm_timur.write([0, 0, 0, 0])
        tm_selatan.write([0,0,0,0])
        tm_barat.write([0,0,0,0])
        tm_utara.write([0,0,0,0])    
        GPIO.cleanup()
    
    finally:
        loop.close()