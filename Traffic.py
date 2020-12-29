class Traffic:

    def __init__(self,arah,greentime,redtime):
        self.__arah = arah
        self.__greentime = greentime
        self.__redtime = redtime
    
    def getArah(self):
        return self.__arah
    
    def getGreenTime(self):
        return self.__greentime
    
    def getRedTime(self):
        return self.__redtime
    
    def setArah(self,arah):
        self.__arah = arah

    def countGreenTime(self,total_kendaraan):
        threshold = 15
        waktu = total_kendaraan*2
        if waktu >= threshold:
            self.__greentime =  threshold
        else:
            self.__greentime = waktu
    
    def setRedTime(self,redtime):
        self.__redtime = redtime
