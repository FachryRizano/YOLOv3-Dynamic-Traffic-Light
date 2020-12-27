import RPi.GPIO as GPIO
import time
import datetime
import tm1637
# from pynput import keyboard  # using module keyboard

GPIO.setwarnings(False)


# [green,yellow,red]
ruas={
    'timur':[22,27,17],
    'selatan':[21,20,16],
    'barat':[7,8,25],
    'utara':[18,15,14]
    }
# green = 22
# yellow = 27
# red = 17


#set up numbering scheme
GPIO.setmode(GPIO.BCM)
# GPIO.setmode(GPIO.BOARD)

# #setup 7 segment
# Display = tm1637.TM1637(22,5,tm1637.BRIGHT_TYPICAL)
# Display.Clear()
# Display.SetBrightnes(1)

# GPIO.setup(ruas['timur'][0],GPIO.OUT)
# GPIO.setup(ruas['timur'][1],GPIO.OUT)
# GPIO.setup(ruas['timur'][2],GPIO.OUT)

# GPIO.setup(ruas['selatan'][0],GPIO.OUT)
# GPIO.setup(ruas['selatan'][1],GPIO.OUT)
# GPIO.setup(ruas['selatan'][2],GPIO.OUT)

# GPIO.setup(ruas['barat'][0],GPIO.OUT)
# GPIO.setup(ruas['barat'][1],GPIO.OUT)
# GPIO.setup(ruas['barat'][2],GPIO.OUT)

# GPIO.setup(ruas['utara'][0],GPIO.OUT)
# GPIO.setup(ruas['utara'][1],GPIO.OUT)
# GPIO.setup(ruas['utara'][2],GPIO.OUT)

for arah in ruas.values():
    # for lampu in range(0,len(arah)):
        # GPIO.setup(arah[lampu],GPIO.OUT)
    GPIO.setup(arah[0],GPIO.OUT)
    GPIO.setup(arah[1],GPIO.OUT)
    GPIO.setup(arah[2],GPIO.OUT)

# set tm1637
tm = tm1637.TM1637(clk=ruas['timur'][0],dio=5)
tm_2 = tm1637.TM1637(clk=ruas['selatan'][0],dio=6)
tm_3 = tm1637.TM1637(clk=ruas['barat'][0],dio=13)
tm_4 = tm1637.TM1637(clk=ruas['utara'][0],dio=19)

#initially turn 

try:

    while(True):
        tm.show("JANO")
        tm_2.show('WAS')
        tm_3.show("HERE")
        tm_4.show('----')

        time.sleep(0.1)
        GPIO.output(ruas['timur'][0],False) 
        GPIO.output(ruas['timur'][0],True)
        time.sleep(0.1)
        GPIO.output(ruas['timur'][0],False)
        GPIO.output(ruas['timur'][1],True)
        time.sleep(0.1)
        GPIO.output(ruas['timur'][1],False)
        GPIO.output(ruas['timur'][2],True)
        time.sleep(0.1)
        GPIO.output(ruas['timur'][2],False)
        
        time.sleep(0.1)
        GPIO.output(ruas['selatan'][0],False)
        GPIO.output(ruas['selatan'][0],True)
        time.sleep(0.1)
        GPIO.output(ruas['selatan'][0],False)
        GPIO.output(ruas['selatan'][1],True)
        time.sleep(0.1)
        GPIO.output(ruas['selatan'][1],False)
        GPIO.output(ruas['selatan'][2],True)
        time.sleep(0.1)
        GPIO.output(ruas['selatan'][2],False)
        
        time.sleep(0.1)
        GPIO.output(ruas['barat'][0],False)
        GPIO.output(ruas['barat'][0],True)
        time.sleep(0.1)
        GPIO.output(ruas['barat'][0],False)
        GPIO.output(ruas['barat'][1],True)
        time.sleep(0.1)
        GPIO.output(ruas['barat'][1],False)
        GPIO.output(ruas['barat'][2],True)
        time.sleep(0.1)
        GPIO.output(ruas['barat'][2],False)
        
        time.sleep(0.1)
        GPIO.output(ruas['utara'][0],False)
        GPIO.output(ruas['utara'][0],True)
        time.sleep(0.1)
        GPIO.output(ruas['utara'][0],False)
        GPIO.output(ruas['utara'][1],True)
        time.sleep(0.1)
        GPIO.output(ruas['utara'][1],False)
        GPIO.output(ruas['utara'][2],True)
        time.sleep(0.1)
        GPIO.output(ruas['utara'][2],False)
        
        
# GPIO.output(yellow,False)
except KeyboardInterrupt:
    tm.write([0, 0, 0, 0])
    tm_2.write([0,0,0,0])
    tm_3.write([0,0,0,0])
    tm_4.write([0,0,0,0])    
    GPIO.cleanup()
        # Display.Clear()
    # break
        # pass