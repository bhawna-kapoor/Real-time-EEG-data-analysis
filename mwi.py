from pylsl import StreamInlet, resolve_streams
import mne 

channel_names = ["Fp1",       "Fpz",       "Fp2",       "F7",        "F3",        "Fz",        "F4",
    "F8",        "FC5",       "FC1",       "FC2",       "FC6",       "M1",        "T7",
    "C3",        "Cz",        "C4",        "T8",        "M2",        "CP5",       "CP1",
    "CP2",       "CP6",       "P7",        "P3",        "Pz",        "P4",        "P8",
    "POz",       "O1",        "O2",        "EOG",       "AF7",       "AF3",       "AF4",
    "AF8",       "F5",        "F1",        "F2",        "F6",        "FC3",       "FCz",
    "FC4",       "C5",        "C1",        "C2",        "C6",        "CP3",       "CP4",
    "P5",        "P1",        "P2",        "P6",        "PO5",       "PO3",       "PO4",
    "PO6",       "FT7",       "FT8",       "TP7",       "TP8",       "PO7",       "PO8",
    "OZ"]

print(len(channel_names))

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
        while True:
            # get a new sample
            eeg_data, timestamps = inlet.pull_sample(timeout=2)
            print(timestamps, eeg_data)

            data_info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg', verbose=None)
            raw = mne.io.RawArray(data=eeg_data, info=data_info).load_data()

            # downsampling raw data
            downsampling_freq = 256
            downsampled_data = mne.io.Raw.resample(raw, sfreq=downsampling_freq)

            # bandpass filtering the data
            bandpass_fil_data = mne.io.Raw.filter(downsampled_data, l_freq=1, h_freq=40)

