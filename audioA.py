import numpy as np
import matplotlib.pyplot as plt
import librosa 
from scipy.signal import find_peaks
from color import bcolors
import pandas as pd
PATH = "impact_audio/object #1 audio trial cut.mp3"


#frequency domain, we look at all the frequency components of a sound and we see how much
# they contribute to the overall signal


#timeSeries is an np array of floating point number constrained [-1.0,1.0]

def rms(n):
    return np.sqrt(np.sum(n**2)/len(n))
def load_file(path, offset, duration):
    timeSeries, sampleRate = librosa.load(path)

    
    maxTime = timeSeries.size/sampleRate
    print(maxTime)
    timeSteps  = np.linspace(0, maxTime, timeSeries.size)
    return timeSteps, timeSeries, sampleRate




def global_max(amplitude_db):
    return np.argmax(amplitude_db)

def findPeaks(amplitude_db):
    peaks_ind, properties = find_peaks(amplitude_db)
    return peaks_ind

def frequency_spectrum(timeSeries,samplingRate,initialTime, finalTime):
    #hann window creates a window in which the endpoints of a signal approach zeor, this is
    #to minimize spectral leakage as well as making the singal periodic for better DFT analysis

    
    s_i = initialTime * samplingRate
    s_f = finalTime * samplingRate

    segment = timeSeries[s_i:s_f]

    dft_input = timeSeries[s_i:s_f]
    window = np.hanning(len(dft_input))
    windowed_input = dft_input*window
    dft = np.fft.rfft(windowed_input)

    amplitude = np.abs(dft)
    dbs_to_linear = 10**(-59.2276/20)
    #sets the reference level to the maximum amplitude in your spectrogram. Any value equal to this will be set to 0 db, being the loudest
    amplitude_db = librosa.amplitude_to_db(amplitude, ref = dbs_to_linear)


    frequencies = np.fft.rfftfreq(len(dft_input),1/samplingRate)

    return frequencies, amplitude_db, segment

"""
the amplitude valuesrange from [-1, 1 ] floating type

"""
def main():

    time, amplitude, sr = load_file(path = PATH,offset=0,duration=5)
    audio_duration = len(amplitude)/sr
    print(time.size)

    interval_1_ini = 2
    interval_1_end = min(3, audio_duration)

    interval_2_ini = 6
    interval_2_end = 11


    print("audio duration : " , audio_duration)

    frequency_1, amplitude_db_1,segment_1 = frequency_spectrum(amplitude,sr, initialTime=interval_1_ini, finalTime=interval_1_end)
    frequency_2, amplitude_db_2, segment_2 = frequency_spectrum(amplitude,sr, initialTime=interval_2_ini, finalTime=interval_2_end)
    complete_freq, amplitude_db_complete, full_segment = frequency_spectrum(amplitude,sr, initialTime=0, finalTime=16)
    
    # print(global_max(amplitude_db_1))

    #at what frequency do we have the maximum ampliude? (db)
    idx = global_max(amplitude_db_1)
    idx2 = global_max(amplitude_db_2)
   
    peak_1 = np.max(amplitude_db_1)
    peak_2 = np.max(amplitude_db_2)

    text = """"
    segment 1 | max intensity [2,3] : {peak_1} , frequency at which the max amplitude in db has been recorded : {freq_1}\n
    segment 2 | max intensity [6,11]: {peak_2} , frequency at which the max amplitude in db has been recorded : {freq_2}\n
    
    """.format(peak_1=peak_1,freq_1=frequency_1[idx],peak_2=peak_2,freq_2=frequency_2[idx2])
    print(text)


    
    print(f"linear rms : {rms(segment_1)}, db rms :   {-20*np.log10(rms(segment_1[:]))}")
    print(f"linear rms : {rms(segment_2)}, db rms :   {-20*np.log10(rms(segment_2[:]))}")


    fig, axs = plt.subplots(nrows = 5,ncols=1, layout = "constrained")
    axs[0].axvspan(interval_1_ini, interval_1_end, alpha=0.5, color='#000000')
    axs[0].axvspan(interval_2_ini, interval_2_end, alpha=0.5, color='#000000')
    axs[0].plot(time, amplitude, color = "#ff000d")
   
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(time[interval_1_ini*sr:interval_1_end*sr], amplitude[interval_1_ini*sr:interval_1_end*sr],color="#ff000d")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Waveform [2,3]")
    
    axs[2].set_title(f"Frequency domain on interval {[interval_1_ini,interval_1_end]} (s)")
    axs[2].plot(frequency_1, amplitude_db_1, color="#0e8705")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Amplitude (db)")
    axs[2].set_xscale("log")

    axs[3].set_title(f"Frequency domain on interval {[interval_2_ini,interval_2_end]} (s)")
    axs[3].plot(frequency_2, amplitude_db_2, color="#0d00ff")
    axs[3].set_xlabel("Frequency (Hz)")
    axs[3].set_ylabel("Amplitude (db)")
    axs[3].set_xscale("log")


    axs[4].set_title(f"Frequency domain on the entire interval")
    axs[4].plot(complete_freq, amplitude_db_complete, color="#dc47d2")
    axs[4].set_xlabel("Frequency (Hz)")
    axs[4].set_ylabel("Amplitude (db)")
    axs[4].set_xscale("log")

    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    axs[4].grid(True)

    plt.show()
if __name__ == "__main__":
    main()
