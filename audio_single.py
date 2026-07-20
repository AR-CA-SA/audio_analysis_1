import numpy as np
import matplotlib.pyplot as plt
import librosa 
from scipy.signal import find_peaks
from color import bcolors
import pandas as pd
PATH = "impact_audio/Object #3 audio trialcut.mp3"

impact_noise = {"Object": [],
                "Material":[],
                "Dimensions": [],
                "DF_1": [],
                "DF_2": [],
                "DF_3": [],
                "Intensity_1":[],
                "Intensity_2":[],
                "Intensity_3":[]}

impact_noise_pd = pd.DataFrame(impact_noise)
#frequency domain, we look at all the frequency components of a sound and we see how much
# they contribute to the overall signal


#timeSeries is an np array of floating point number constrained [-1.0,1.0]


def rms(n):
    return np.sqrt(np.sum(n**2)/len(n))
def load_file(path):
    timeSeries, sampleRate = librosa.load(path)

    
    maxTime = timeSeries.size/sampleRate
    print(maxTime)
    timeSteps  = np.linspace(0, maxTime, timeSeries.size)
    return timeSteps, timeSeries, sampleRate




def global_max(amplitude_db):
    return np.argmax(amplitude_db)

def findPeaks(amplitude_db):
    peaks_ind, _ = find_peaks(amplitude_db)
    return peaks_ind

def frequency_spectrum(timeSeries,samplingRate):
    #hann window creates a window in which the endpoints of a signal approach zeor, this is
    #to minimize spectral leakage as well as making the singal periodic for better DFT analysis

    
    segment = timeSeries

    dft_input = timeSeries[:]
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

    time, amplitude, sr = load_file(path = PATH)
    audio_duration = len(amplitude)/sr
    print(time.size)

    print("audio duration : " , audio_duration)

    frequency_1, amplitude_db_1,segment_1 = frequency_spectrum(amplitude,sr)
    complete_freq, amplitude_db_complete, full_segment = frequency_spectrum(amplitude,sr )
    
    # print(global_max(amplitude_db_1))

    #at what frequency do we have the maximum ampliude? (db)
    idx = global_max(amplitude_db_1)



    peak_1 = np.max(amplitude_db_1)


    text = """"
    full audio | max intensity [2,3] : {peak_1} , frequency at which the max amplitude in db has been recorded : {freq_1}\n
    
    
    """.format(peak_1=peak_1,freq_1=frequency_1[idx])
    print(text)


    
    print(f"linear rms : {rms(segment_1)}, db rms :   {-20*np.log10(rms(segment_1[:]))}")



    fig, axs = plt.subplots(nrows = 2,ncols=1, layout = "constrained")
    # axs[0].axvspan(interval_1_ini, interval_1_end, alpha=0.5, color='#000000')
    # axs[0].axvspan(interval_2_ini, interval_2_end, alpha=0.5, color='#000000')
    axs[0].plot(time, amplitude, color = "#ff000d")
   
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(time, amplitude,color="#ff000d")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Waveform [2,3]")

    axs[0].grid(True)
    axs[1].grid(True)

    

    plt.show()
if __name__ == "__main__":
    main()
