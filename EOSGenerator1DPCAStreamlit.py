import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    return np.gradient(y, x, edge_order=2)

def compute_energy_density(T, P):
    """This function computes energy density"""
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P    # energy density
    return e

def compute_speed_of_sound_square(T, P):
    """This function computes the speed of sound square"""
    e = compute_energy_density(T, P)
    dPde = compute_derivative(e, P)
    return dPde

@st.cache_resource
def loadPCA(pcaFile):
    with open(pcaFile, 'rb') as f:
        pca = pickle.load(f)
    return pca

@st.cache_resource
def loadHotQCD(hotQCDFile):
    hotQCD = np.loadtxt(hotQCDFile)
    return hotQCD[::7, :]

def main():
    pcaFile = "EoSPCA.pickle"
    pca = loadPCA(pcaFile)

    hotQCDFile = "EoS_hotQCD.dat"
    hotQCD = loadHotQCD(hotQCDFile)

    st.sidebar.header('Model Parameters:')
    params = []     # record the model parameter values
    for iPC in range(pca.n_components):
        parVal = st.sidebar.slider(label=f"PC: {iPC}",
                                   min_value=round(pca.pcMin[iPC], 2),
                                   max_value=round(pca.pcMax[iPC], 2),
                                   value=0.,
                                   step=(pca.pcMax[iPC] - pca.pcMin[iPC])/1000.,
                                   format='%f')
        params.append(parVal)
    params = np.array([params,])

    T_plot = np.exp(np.linspace(np.log(0.006), np.log(0.9447), 1000))
    EOS_P = pca.inverse_transform(params).flatten()

    EOS_cs2 = compute_speed_of_sound_square(T_plot, EOS_P)
    hotQCD_cs2 = compute_speed_of_sound_square(
        hotQCD[:, 0], hotQCD[:, 1]*(hotQCD[:, 0]**4))

    fig = plt.figure()
    plt.plot(T_plot, EOS_P/(T_plot**4), '-r')
    plt.plot(hotQCD[::3, 0], hotQCD[::3, 1], 'ob', label="hotQCD")
    plt.legend()
    plt.xlim([0, 0.6])
    plt.ylim([0, None])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$P/T^{4}$")
    st.pyplot(fig)

    fig = plt.figure()
    plt.plot(T_plot, EOS_cs2, '-r')
    plt.plot(hotQCD[::3, 0], hotQCD_cs2[::3], 'ob', label="hotQCD")
    plt.legend()
    plt.xlim([0, 0.6])
    plt.ylim([0, 0.4])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$c_s^2$")
    st.pyplot(fig)


if __name__ == '__main__':
    main()
