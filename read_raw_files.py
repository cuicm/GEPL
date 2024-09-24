import mne
import os
import numpy as np
import pickle
import scipy

def read_raw_tinnus(filepath):
    file_list = os.listdir(filepath)
    eeg_data = []
    for file in file_list:
        if '.set' not in file:
            continue
        file_path=f"{filepath}\{file}"

        raws = mne.io.read_epochs_eeglab(file_path,uint16_codec='ascii')
        data = raws.get_data().astype(np.float32)
        data = np.concatenate(data, axis=1)
        data = data * 1e6
        eeg_data.append(data)
    return eeg_data

def read_raw_deap(filepath='RAW\DEAP Dataset'):

    def read_eeg_signal_from_file(filename):
        x = pickle._Unpickler(open(filename, 'rb'))
        x.encoding = 'latin1'
        p = x.load()
        return p

    # Load only 22/32 participants with frontal videos recorded
    files = []
    for n in range(1, 23): 
        s = ''
        if n < 10:
            s += '0'
        s += str(n)
        files.append(s)
    print(files)

    # 22x40 = 880 trials for 22 participants
    labels = []
    data = []

    for i in files: 
        filename = f"{filepath}\data_preprocessed_python\s" + i + ".dat"
        trial = read_eeg_signal_from_file(filename)
        labels.append(trial['labels'])
        data.append(trial['data'])

    labels = np.array(labels)
    labels = labels.flatten()
    labels = labels.reshape(880, 4)

    data = np.array(data)
    data = data.flatten()
    data = data.reshape(880, 40, 8064)

    print(labels)

    eeg_channels = np.array(["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"])
    peripheral_channels = np.array(["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"])

    eeg_data = []
    for i in range (len(data)):
        eeg_data.append(data[i][:len(eeg_channels)])
    return eeg_data


def read_raw_shl(filepath):
    
    def read_raw_files(filepath,eeg_data,labels,label):
        eletrodes=[1, 2, 3, 5, 8, 10, 12, 14, 18, 21, 22, 23, 25, 26, 27, 28, 29, 32, 33, 35, 36, 40, 41, 44, 45, 46, 50, 51, 57, 61, 64, 66, 69, 71, 74, 76, 82, 86, 89, 91, 92, 95, 96, 97, 101, 102, 103, 104, 107, 110, 111, 115, 116, 121, 122, 123, -1]
        file_list = os.listdir(filepath)
        for file in file_list:
            if '.set' not in file:
                continue
            file_path=f"{filepath}\{file}"

            if 'CN' in file_path:
                raws = mne.io.read_raw_eeglab(file_path,uint16_codec='latin1')
                data,times = raws[:]
                data=data[eletrodes]
                labels.append(0)
            else:
                raws = mne.io.read_epochs_eeglab(file_path,uint16_codec='latin1')
                # _mio5 line 204 : uint16_codec='latin1' 
                data = raws.get_data()
                data = np.concatenate(data, axis=1)
                labels.append(1)
            data = data * 1e6
            eeg_data.append(data)
    eeg_data = []
    labels = []
    read_raw_files(f'{filepath}\CN',eeg_data,labels,0)
    read_raw_files(f'{filepath}\SHL',eeg_data,labels,1)
    return eeg_data,labels

def read_raw_alzh(filepath):

    file_list = os.listdir(filepath)
    eeg_data = []
    labels = [1]*36 + [0]*29 + [2]*23
    for file in file_list:
        if 'sub-' not in file:
            continue
        file_path = f'{filepath}\{file}\eeg\{file}_task-eyesclosed_eeg.set'
        print(file_path)
        raw = mne.io.read_raw_eeglab(file_path,preload=False)
        data,times=raw[:]
        data = data * 1e6
        eeg_data.append(data)
    return eeg_data,labels


def read_raw_TE(filepath):

    def read_raw_files(filepath,eeg_data,labels,label):
        sample_length = 500 * 50 # sample rate * second
        file_list = os.listdir(filepath)
        for file in file_list:
            mat_file_path = f'{filepath}\{file}'
            print(mat_file_path)
            mat_contents = scipy.io.loadmat(mat_file_path)

            data = mat_contents['data']

            sampling_times, electrode_count = data.shape
            signal_length = data[0, 0].shape[0]

            result_array = np.zeros((electrode_count-1, sampling_times * signal_length), dtype=np.float32)

            for electrode_index in range(electrode_count-1):
                concatenated_signals = np.concatenate([data[time_index, electrode_index].squeeze() for time_index in range(sampling_times)])
                result_array[electrode_index, :] = concatenated_signals

            max_start_index = result_array.shape[1] - sample_length
            start_index = np.random.randint(0, max_start_index)
            result_array = result_array[:, start_index:start_index + sample_length]

            eeg_data.append(result_array)
            labels.append(label)

    eeg_data = []
    labels = []
    read_raw_files(f'{filepath}\epilepsy',eeg_data,labels,0)
    read_raw_files(f'{filepath}\\normal',eeg_data,labels,1)
    
    return eeg_data,labels


