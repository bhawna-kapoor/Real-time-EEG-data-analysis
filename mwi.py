from pylsl import StreamInlet, resolve_streams
import mne
import pandas

channel_names = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7",
                 "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
                 "POz", "O1", "O2", "EOG", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FC3", "FCz",
                 "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4", "P5", "P1", "P2", "P6", "PO5", "PO3", "PO4", 
                 "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "OZ"]

# print(len(channel_names))

print("Looking for EEG streams...")
streams = resolve_streams(wait_time=3)

for stream in streams:
    inlet = StreamInlet(stream)
    info = inlet.info()
    name = info.name()
    # checking if the stream captured is our stream
    if name == 'openvibeSignal':
        fs = float(info.nominal_srate())
        print("Processing stream...")
        try:
            while True:
                # get a new sample
                eeg_data, timestamps = inlet.pull_chunk(timeout=2, max_samples=64)
                print(timestamps, eeg_data)

                data_info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg', verbose=None)
                raw = mne.io.RawArray(data=eeg_data, info=data_info).load_data()

                # downsampling raw data
                downsampling_freq = 256
                downsampled_data = mne.io.Raw.resample(raw, sfreq=downsampling_freq)

                # bandpass filtering the data
                bandpass_fil_data = mne.io.Raw.filter(downsampled_data, l_freq=1, h_freq=40,
                                                      picks=['F3', 'F4', 'P7', 'P8'])

                # Calculation of Relative Power
                # Relative Power = power of certain frequency band / summation of all frequency bands power
                
                # all frequency bands power at selected channels 
                f3_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=1, fmax=40,
                                               picks='F3').to_data_frame()
                f4_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=1, fmax=40,
                                               picks='F4').to_data_frame()
                p7_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=1, fmax=40,
                                               picks='P7').to_data_frame()
                p8_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=1, fmax=40,
                                               picks='P8').to_data_frame()

                f3_mean = f3_df['F3'].mean()
                f4_mean = f4_df['F4'].mean()
                p7_mean = p7_df['P7'].mean()
                p8_mean = p8_df['P8'].mean()

                # power of required frequency band with selected channels
                f3_theta_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=0, fmax=4,
                                                     picks='F3').to_data_frame()
                f4_theta_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=0, fmax=4,
                                                     picks='F4').to_data_frame()
                p7_alpha_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=8, fmax=12,
                                                     picks='P7').to_data_frame()
                p8_alpha_df = mne.io.Raw.compute_psd(bandpass_fil_data, method='welch', fmin=8, fmax=12,
                                                     picks='P8').to_data_frame()

                f3_theta_mean = f3_theta_df['F3'].mean()
                f4_theta_mean = f4_theta_df['F4'].mean()
                p7_alpha_mean = p7_alpha_df['P7'].mean()
                p8_alpha_mean = p8_alpha_df['P8'].mean()

                # Formula for relative power used
                f3_theta_rp = f3_theta_mean / f3_mean
                f4_theta_rp = f4_theta_mean / f4_mean
                p7_alpha_rp = p7_alpha_mean / p7_mean
                p8_alpha_rp = p8_alpha_mean / p8_mean

                # Calculating mental workload index
                # Mental workload index = Average frontal theta power / Average parietal alpha power
                # (F3 theta RP + F4 theta RP) / (P7 Alpha RP + P8 Alpha RP)
                avg_frontal_theta_power = f3_theta_rp + f4_theta_rp
                avg_parietal_alpha_power = p7_alpha_rp + p8_alpha_rp
                mwi = avg_frontal_theta_power / avg_parietal_alpha_power
                print(f"mwi: {mwi}")
        except KeyboardInterrupt:
            print("Closing")

