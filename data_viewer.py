import sys
from matplotlib import pyplot, colors
import numpy
from scipy.fft import fft, fftfreq
from scipy.interpolate import LinearNDInterpolator
import scipy.signal as signal
from numpy import ndarray
import numpy as np
import re

from tools.timer import timer
t=timer.time
s=timer.set

SAMPLES_PER = 51200
MAX_ROWS_TO_IMPORT = 100000

file_list = []
def main() :
    s()
    
    # Usage
        # python data_viewer.py <plot_option> <data modification option for file1> <file1> <data modification option for file2> <file2> ... 
    
    try :
        plot_options = sys.argv[-1]
        
        trimmed_df_list = sys.argv[1:-1]
        data_file_list = []

        for i in range(0, len(trimmed_df_list), 2) :
            data_file_list.append([trimmed_df_list[i], trimmed_df_list[i+1]] )
    except:
        print("Incorrect Usage: \n python data_viewer.py <plot_option> <data modification option for file1> <file1> <data modification option for file2> <file2> ...")
    print(data_file_list)

    print("")
    t("data load")

    #Additional files input to plot at the same time all with the same options
    data = []
    t("data import")
    for ic, [data_options, file] in enumerate(data_file_list) :
        loaded_file:ndarray = numpy.genfromtxt(file, delimiter=',', skip_header=1, max_rows=MAX_ROWS_TO_IMPORT)

        # Data Modification Options
        if("t" in data_options) : 
            print('delayed')
            pattern = r't(\d+)'
            delay = int(re.search(pattern, data_options).group(1))
            loaded_file = loaded_file[delay::]

        if("d5" in data_options) : 
            print('amplified')
            pattern = r'd(\d+)'
            factor = int(re.search(pattern, data_options).group(1))
            print(loaded_file)
            loaded_file = loaded_file/factor
            # numpy.savetxt(f"{file}_dived", loaded_file, delimiter=",")
            # print(loaded_file)

        
        # data.append(loaded_file)

        # if ("-" in data_options) : data.append(loaded_file)
        # print(loaded_file)


        if("a" in data_options) : data.extend([d1, ch_1])
        if(("r" not in data_options) and ("0" not in data_options) and ("1" not in data_options) and ("d" not in data_options)) : data.extend([d1, ch_1])
        if("d" in data_options) : data.append(ch_1-d1) #this order to maintain backwards compatibiity
        if("IZ" in data_options) : data.append(interspace_data(d1, ch_1))
        if("IO" in data_options) : data.append(interspace_data(ch_1, d1))
        if("m" in data_options) : data.append(remove_dom_freq(ch_0))
        if("c" in data_options) : data.extend([decimate(d1), decimate(ch_1)])

        # if("p" in data_options) : 
        #     d = ch_1-d1
        #     mean = numpy.average(d)
        #     # SR = 51.2*1000 #S/s
        #     # F = 119

        #     gain = 0
        #     data.append(d-mean-gain)

        # if("e" in data_options) : 
        #     a,b = add_delay(ch_0, ch_1, 1)
        #     data.append(b-a)
        if("s" in data_options) : data.extend([ch_0[0::5] ])


    import numpy as np
    # Visualisation Options
    if("d" in plot_options) : 
    # Determine which array is larger
        

    # Subtract the smaller array from the concatenated array
        data = [(data[0] - np.resize(data[1], (87061,3)))]
        
        
    if("p" in plot_options) : plot(data)
    if("h" in plot_options) : histagram(data)
    if("t" in plot_options) : fftt(data) #Intensity-Frequency
    if("s" in plot_options) : fftp(data) #Intensity-Phase
    if("c" in plot_options) : 
        s()
        for i in data: colour_plot(i)
  
def concatenate_arrays(a, b):
    """Concatenate the larger array to the size of the smaller one"""
    if a.shape[0] > b.shape[0]:
        larger_array = a
        smaller_array = b
    else:
        larger_array = b
        smaller_array = a

    n = smaller_array.shape[0]
    return np.concatenate((larger_array[:n], larger_array[n:]), axis=0)


# method to interspace data from both channels
def interspace_data(ch_0, ch_1) :
    array = numpy.empty(len(ch_0) + len(ch_1), dtype=float)
    array[0::2] = ch_0
    array[1::2] = ch_1
    return array

def decimate(ch): return signal.decimate(ch, 5)

from math import pi
def remove_dom_freq(ch) :
    xf = fftfreq(len(ch), 1 / SAMPLES_PER)
    yf = fft(ch)

    max_index = numpy.argmax( abs( yf ) ) #Get index of fft with highest frequency in the positive range
    
    print(f"frequency: {xf[max_index]} Intensity: {numpy.abs(yf[max_index])}, phase: {numpy.angle(yf[max_index])}")

    p = numpy.angle(yf[max_index])

    A = numpy.abs(yf[max_index])/len(xf)*2

    w = xf[max_index]*2*pi #Convert to angular frequency

    x = numpy.linspace(0,100, 99999)

    sin_to_subt = A*numpy.sin(w*x + p)
    print(f"{A}sin({w}t + {p})")
    return ch-sin_to_subt

def trim_top(delay:int, data) :
    return [ i[delay::] for i in data ]



GLOBAL_TRANSPARENT = 0.5
# Visualisation methods
def plot(data) :
    for ic, i in enumerate(data):
        pyplot.plot(i, label=f"Data: {ic}", alpha= (1 if len(file_list)==1 else GLOBAL_TRANSPARENT))
    pyplot.title("Plot")
    pyplot.legend(loc='upper right') #Show input Labels
    pyplot.xlabel("Sample Number")
    pyplot.ylabel("Voltage (V)")

    pyplot.show()

l = ["DAQ 1", "DAQ 2"]
def histagram(data, file="Histogram") :
    DYNAMIC_BUCKETS = 1000/(51200*15)
    BUCKETS= 1000
    for ic, i in enumerate(data):
        # weight = numpy.ones(len(i)) / len(i)
        print("std dev",numpy.std(i))
        pyplot.hist(i, int(BUCKETS), label=f"{l[ic]}", alpha= (1 if len(file_list)==1 else GLOBAL_TRANSPARENT) )

    pyplot.xlabel("Voltage Difference")
    pyplot.ylabel("Percentage of Samples")
    pyplot.title(file)
    # pyplot.xlim(-0.0010, 0.0010)
    # pyplot.ylim(0, 0.006)
    pyplot.legend(loc='upper right') #Show input Labels
    pyplot.show()

def fftt(data) :
    #FFT
    pyplot.title("FFT: Frequency vs Power")

    for ic, i in enumerate(data):
        xf = fftfreq(len(i), 1 / SAMPLES_PER)
        # pyplot.plot(xf, numpy.log(abs(fft(i))))
        fft_data = fft(i)
        abs_fft = abs(fft_data)

        pyplot.plot(xf, abs_fft,label=f"Data: {[ic]}", alpha= (1 if len(file_list)==1 else GLOBAL_TRANSPARENT))

        peak_index = numpy.argmax(abs_fft)

        peak_freq = xf[peak_index]
        peak_phase = numpy.angle(fft_data[peak_index])
        peak_power = numpy.abs(fft_data[peak_index])

        print(f"Max peak info: ", peak_freq, peak_phase, peak_power  )

    pyplot.xlabel("Frequency")
    pyplot.ylabel("Power")
    pyplot.legend(loc=7)
    pyplot.show()

def fftp(data):
    pyplot.title("FFT: Phase Shift vs Power")

    for ic, i in enumerate(data):
        ft = fft(i)
        pyplot.plot(numpy.angle(ft), abs(ft), label=f"Data {ic}", marker=None ,alpha= (1 if len(file_list)==1 else GLOBAL_TRANSPARENT))

    pyplot.legend(loc='upper right')
    pyplot.xlim((-3.17, 3.17))
    pyplot.ylim(bottom=-5)
    pyplot.xlabel("Phase")
    pyplot.ylabel("Power")
    pyplot.show()

def colour_plot(ch) :
    
    C_fft_data = fft(ch)
   
    x = fftfreq(len(ch), 1 / SAMPLES_PER) #Frequency
    y = numpy.angle(C_fft_data) #Phase shift
    z = numpy.abs(C_fft_data) #Intensity 

    X = numpy.linspace(0, 5000, 500)
    Y = numpy.linspace(-pi, pi, 60)

    s()
    X, Y = numpy.meshgrid(X, Y)  # 2D grid for interpolation
    t('meshgrid')
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    t('linNdinterp')
    Z = interp(X, Y)
    
    t(f'mass interp {Z.size}')
    
    MAX = Z[Z.argmax()]


    def _forward(d) :
        return d/MAX  
    def _inverse(d) :
        return d*MAX

    # # Log PCT
    # def _forward(d) :
    #     return numpy.log(d/MAX +1)
    # def _inverse(d) :
    #     return ((10**d) -1)*MAX

    # def _forward(d) :
    #     return (d/MAX)**3  
    # def _inverse(d) :
    #     return (d*MAX)**3
    
    class cus_norm(colors.Normalize):
        def _forward(d) :
            return numpy.log(d/MAX +1)
        def _inverse(d) :
            return ((10**d) -1)*MAX

    # pyplot.pcolormesh(X, Y, Z, shading='nearest', cmap='rainbow')
    # pyplot.pcolormesh(X, Y, Z, shading='nearest', cmap='rainbow', norm=colors.LogNorm())
    pyplot.pcolormesh(X, Y, Z, shading='auto', cmap='rainbow', norm=cus_norm())
    t("colour mesh")
    # pyplot.plot(X, Y)
    pyplot.scatter(X, Y, 400, facecolors="none")

    t("scatter")
    pyplot.colorbar()

    pyplot.title("FFT")
    pyplot.xlabel("Frequency (Hz)")
    pyplot.ylabel("Phase Shift (Rads)")

    pyplot.xlim([-5, 5000])
    pyplot.ylim([-pi, pi])
    t("before show")
    pyplot.show()
    t("after show")

# Random tools
from tools.RMSE import RMSE 
def fft_lin_test(ch0, ch1):
    a = fft(ch0) - fft(ch1)
    b = fft((ch0-ch1))
    print("fft linear" , RMSE(a, b), max(abs(a-b)))

if __name__ == '__main__' :
    main()