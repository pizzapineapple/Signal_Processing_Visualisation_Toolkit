from time import perf_counter

class timer :
    init_time = False
    hush = True
    def time(text="time elapsed") :
        if timer.init_time == False or text=="hush":
            timer.init_time = perf_counter()
        elif timer.hush == False: print(f"{text}, {perf_counter()-timer.init_time}")
    def set() :
        timer.init_time = perf_counter()
