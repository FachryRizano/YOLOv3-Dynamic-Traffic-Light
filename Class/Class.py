import RPi.GPIO as GPIO
class Traffic:

    def __init__(self,arah, dio,pinTraffic):
        self.__arah = arah
        self.__dio = dio
        self.__pinTraffic = pinTraffic
        
    def getDio(self):
        return self.__dio

    def getGreenTime(self):
        return self.__greentime
    
    def getRedTime(self):
        return self.__redtime
    
    def countGreenTime(self,total_kendaraan):
        threshold = 15
        waktu = total_kendaraan*2
        if waktu >= threshold:
            return threshold
        else:
            return waktu

    def updateTime(self,time):
        if time == "green":
            self.__greentime -= 1
        elif time == "red":
            self.__redtime -= 1
    
    def setGreenTime(self,greenTime):
        self.__greentime = greenTime

    def setRedTime(self,time):
        self.__redtime = time

    def countRedTime(self,prev):
        return prev.getGreenTime() + prev.getRedTime() + 3

    def getPinTraffic(self):
        return self.__pinTraffic

    def light_on(self,color):
        for i in range(3):
            GPIO.setup(self.getPinTraffic()[i],GPIO.OUT)
        
        if color == "green":
            GPIO.output(self.getPinTraffic()[0],True)
            GPIO.output(self.getPinTraffic()[1],False)
            GPIO.output(self.getPinTraffic()[2],False)
        elif color == "yellow":
            GPIO.output(self.getPinTraffic()[0],False)
            GPIO.output(self.getPinTraffic()[1],True)
            GPIO.output(self.getPinTraffic()[2],False)
        elif color == "red":
            GPIO.output(self.getPinTraffic()[0],False)
            GPIO.output(self.getPinTraffic()[1],False)
            GPIO.output(self.getPinTraffic()[2],True)