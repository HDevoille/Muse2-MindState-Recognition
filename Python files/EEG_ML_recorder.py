from pylsl import resolve_stream, StreamInlet
import pandas as pd
import numpy as np
import os

streams = resolve_stream('type','EEG')
inlet = StreamInlet(streams[0])

def freq_band(data,iter):
    fs = 256

    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
    # # Define EEG bands
    # eeg_bands = {'Delta': (1, 4),
    #             'Theta': (4, 8),
    #             'Alpha': (8, 12),
    #             'Beta': (12, 30),
    #             'Gamma': (30, 45)}

    # # Take the mean of the fft amplitude for each EEG band
    # eeg_band_fft = dict()
    # for band in eeg_bands:  
    #     freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
    #                     (fft_freq <= eeg_bands[band][1]))[0]
    #     eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    # Plot the data (using pandas here cause it's easy)
    # df = pd.DataFrame(columns=['Delta','Theta','Alpha','Beta','Gamma'],index=[iter])
    # df['Delta'] = [eeg_band_fft['Delta']]
    # df['Theta'] = [eeg_band_fft['Theta']]
    # df['Alpha'] = [eeg_band_fft['Alpha']]
    # df['Beta'] = [eeg_band_fft['Beta']]
    # df['Gamma'] = [eeg_band_fft['Gamma']]

    df = pd.DataFrame(fft_vals)

    return df
    
def writer(df,output_path):
    # output_path='Dataset2/F09.csv'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

def main():
    for i in range(7):
        filename = ['Dataset2/R04.csv','Dataset2/R05.csv','Dataset2/R06.csv','Dataset2/R07.csv','Dataset2/R08.csv','Dataset2/R09.csv']
        iter = 0
        df = pd.DataFrame(columns=[0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,
    19 ,20 ,21 ,22 ,23 ,24 ,25 ,26 ,27 ,28 ,29 ,30 ,31 ,32 ,33 ,34 ,35 ,36 ,37 ,38 ,39 ,40 ,41 ,42 ,43 ,44 ,45 ,46 ,47 ,48 ,49 ,
    50 ,51 ,52 ,53 ,54 ,55 ,56 ,57 ,58 ,59 ,60 ,61 ,62 ,63 ,64 ,65 ,66 ,67 ,68 ,69 ,70 ,71 ,72 ,73 ,74 ,75 ,76 ,77 ,78 ,
    79 ,80 ,81 ,82 ,83 ,84 ,85 ,86 ,87 ,88 ,89 ,90 ,91 ,92 ,93 ,94 ,95 ,96 ,97 ,98 ,99 ,100 ,101 ,102 ,103 ,104 ,105 ,
    106 ,107 ,108 ,109 ,110 ,111 ,112 ,113 ,114 ,115 ,116 ,117 ,118 ,119 ,120 ,121 ,122 ,123 ,124 ,125 ,126 ,127 ,128])
        while iter < 181:
            data = [[],[],[],[]]
            n = 0        
            while n < 256:
                sample, timestamp = inlet.pull_sample()
                data[0].append(sample[0])
                data[1].append(sample[1])
                data[2].append(sample[2])
                data[3].append(sample[3])
                n+=1
            #Test : 

            fs = 256
            fft_vals = np.absolute(np.fft.rfft(data[1]))
            fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

            eeg_fft = dict()

            for i in range(len(fft_freq)):
                eeg_fft[i] = fft_vals[i] 

            df.loc[len(df)] = fft_vals 
            iter += 1
            print(iter)
            # df = freq_band(data[1],iter)
            # writer(df)
            # iter += 1
            # print(iter)
        writer(df,filename[i])
main()