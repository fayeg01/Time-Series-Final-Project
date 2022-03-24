import numpy as np
import emd

def create_raw_dict(training_healthy_samples, training_seizure_samples):
    """
    Returns the raw dictionary from training samples
    """
    healthy_dict = []
    seizure_dict = []
    for sample in training_healthy_samples:
        imfs = emd.sift.sift(sample)
        imfs = imfs.T
        for imf in imfs:
            healthy_dict.append([imf/np.linalg.norm(imf)])
    for sample in training_seizure_samples:
        imfs = emd.sift.sift(sample)
        imfs = imfs.T
        for imf in imfs:
            seizure_dict.append([imf/np.linalg.norm(imf)])
    healthy_dict = np.array(healthy_dict)
    healthy_dict = np.squeeze(healthy_dict, 1)
    seizure_dict = np.array(seizure_dict)
    seizure_dict = np.squeeze(seizure_dict, 1)
    print(f"{healthy_dict.shape[0]} IMFs for healthy dictionary")
    print(f"{seizure_dict.shape[0]} IMFs for seizure dictionary")
    return healthy_dict, seizure_dict

def termination(iter, max_iter):
    """
    Returns True if the termination condition has been reached
    """
    if iter >= max_iter:
        return True
    return False

def dict_learning(X_train, labels_train, D_raw, max_iter=100):
    Xc_train = X_train.copy()
    Xc_healthy = []
    Xc_seizure = []
    for i, label in enumerate(labels_train):
        if label == 0:
            Xc_healthy.append(Xc_train[i])
        else:
            Xc_seizure.append(Xc_train[i])
    Dc_raw = D_raw.copy()
    K_healthy, N = Xc_healthy.shape
    K_seizure, N = Xc_seizure.shape
    Dc_train = []
    for i in range(int(max_iter/2)):
        Xc = Xc_healthy[i % K_healthy]
        alphas = []
        for atom in Dc_raw:
            alphas.append(np.abs(Xc.dot(atom)))
        best_index = np.argmax(alphas)
        Dc_train.append(Dc_raw[best_index].copy())
        rx = Xc - alphas[best_index] * Dc_raw[best_index]
        Dc_raw[best_index] = np.zeros_like(Dc_raw[best_index])
        Xc_healthy[i % K_healthy] = rx
        Xc = Xc_seizure[i % K_seizure]
        alphas = []
        for atom in Dc_raw:
            alphas.append(np.abs(Xc.dot(atom)))
        best_index = np.argmax(alphas)
        Dc_train.append(Dc_raw[best_index].copy())
        rx = Xc - alphas[best_index] * Dc_raw[best_index]
        Dc_raw[best_index] = np.zeros_like(Dc_raw[best_index])
        Xc_seizure[i % K_seizure] = rx
        if termination(i, max_iter):
            break
    return Dc_train

def dict_learningV2(X_train, labels_train, D_raw, max_iter=100):
    Xc_train = X_train.copy()
    Xc_healthy = []
    Xc_seizure = []
    for i, label in enumerate(labels_train):
        if label == 0:
            Xc_healthy.append(Xc_train[i])
        else:
            Xc_seizure.append(Xc_train[i])
    Dc_raw = D_raw.copy()
    K_healthy, N = Xc_healthy.shape
    K_seizure, N = Xc_seizure.shape
    Dc_train = []
    for i in range(int(max_iter/2)):
        alphas = np.zeros(Dc_raw.shape[0])
        for i in range(75):
            Xc = Xc_healthy[i % K_healthy]
            for atom in Dc_raw:
                alphas[i] += np.abs(Xc.dot(atom))
        best_index = np.argmax(alphas)
        Dc_train.append(Dc_raw[best_index].copy())
        for i in range(75):
            Xc = Xc_healthy[i % K_healthy]
            rx = Xc - alphas[best_index] * Dc_raw[best_index]
            Xc_healthy[i % K_healthy] = rx
        Dc_raw[best_index] = np.zeros_like(Dc_raw[best_index])

        alphas = np.zeros(Dc_raw.shape[0])
        for i in range(75):
            Xc = Xc_seizure[i % K_seizure]
            for atom in Dc_raw:
                alphas[i] += np.abs(Xc.dot(atom))
        best_index = np.argmax(alphas)
        Dc_train.append(Dc_raw[best_index].copy())
        for i in range(75):
            Xc = Xc_seizure[i % K_seizure]
            rx = Xc - alphas[best_index] * Dc_raw[best_index]
            Xc_seizure[i % K_seizure] = rx
        Dc_raw[best_index] = np.zeros_like(Dc_raw[best_index])
        
        if termination(i, max_iter):
            break
    return Dc_train

def create_feature_sample(sample, D_train):
    features = []
    for element in D_train:
        features.append(sample.dot(element))
    return features

def create_feature_dataset(samples, D_train):
    features = []
    for sample in samples:
        features.append(create_feature_sample(sample, D_train))
    return np.array(features)