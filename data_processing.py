import numpy as np

def collect_samples_chb01(raw_data_array, channel_to_select=0, limit_of_samples = 1000000):
    """
    Converts an array of raw_data for chb01 patient and
    returns two arrays of healthy and seizure samples
    """
    raw_data_1  = raw_data_array[0]
    raw_data_2  = raw_data_array[1]
    raw_data_3  = raw_data_array[2]
    raw_data_4  = raw_data_array[3]
    raw_data_5  = raw_data_array[4]
    raw_data_6  = raw_data_array[5]
    raw_data_7  = raw_data_array[6]
    raw_data_8  = raw_data_array[7]
    raw_data_9  = raw_data_array[8]
    raw_data_10 = raw_data_array[9]
    raw_data_15 = raw_data_array[10]
    raw_data_16 = raw_data_array[11]
    raw_data_18 = raw_data_array[12]
    raw_data_21 = raw_data_array[13]
    raw_data_26 = raw_data_array[14]
    healthy_samples = []
    seizure_samples = []
    for i in range(1800):
        if not len(healthy_samples) >= limit_of_samples:
            healthy_samples.append(raw_data_1[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_2[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_5[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_6[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_7[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_8[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_9[channel_to_select,i*1024:(i+1)*1024])
            healthy_samples.append(raw_data_10[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 749) and (i < 759):
            seizure_samples.append(raw_data_3[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_3[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 367) and (i < 373):
            seizure_samples.append(raw_data_4[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_4[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 433) and (i < 443):
            seizure_samples.append(raw_data_15[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_15[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 254) and (i < 266):
            seizure_samples.append(raw_data_16[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_16[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 430) and (i < 452):
            seizure_samples.append(raw_data_18[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_18[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 82) and (i < 105):
            seizure_samples.append(raw_data_21[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_21[channel_to_select,i*1024:(i+1)*1024])
    for i in range(1800):
        if (i >= 466) and (i < 490):
            seizure_samples.append(raw_data_26[channel_to_select,i*1024:(i+1)*1024])
        else:
            if not len(healthy_samples) >= limit_of_samples:
                healthy_samples.append(raw_data_26[channel_to_select,i*1024:(i+1)*1024])
    healthy_samples = np.array(healthy_samples)
    seizure_samples = np.array(seizure_samples)
    print(f"{healthy_samples.shape[0]} healthy samples")
    print(f"{seizure_samples.shape[0]} seizure samples")
    return healthy_samples, seizure_samples

def split_train_test_chb01(healthy_samples, seizure_samples):
    """
    Split samples into training and test dataset
    """
    training_healthy_samples = []
    for i in range(75):
        training_healthy_samples.append(healthy_samples[i])
    training_healthy_samples = np.array(training_healthy_samples)
    training_seizure_samples = seizure_samples[:75]
    test_healthy_samples = []
    for i in range(75, 150):
        test_healthy_samples.append(healthy_samples[i])
    test_healthy_samples = np.array(test_healthy_samples)
    test_seizure_samples = seizure_samples[75:]
    return training_healthy_samples, training_seizure_samples, test_healthy_samples, test_seizure_samples

def normalize_data(data):
    """
    Normalize each element of the data array
    """
    for i in range(data.shape[0]):
        data[i] = data[i] / np.amax(data)