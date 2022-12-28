import streamlit as st
import joblib
import pandas as pd
import numpy as np
import librosa as lr
import noisereduce as nr
from tempfile import NamedTemporaryFile
from io import BytesIO
import os
import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")

st_audiorec = components.declare_component("st_audiorec", path=build_dir)

FRAME_SIZE=1024
HOP_LENGTH=512 

st.title("Baby Cry Reason Prediction")

collected_data = pd.DataFrame(columns = ['Amplitude_Envelope_Mean','RMS_Mean', 'ZCR_Mean', 'STFT_Mean', 'SC_Mean', 'SBAN_Mean', 'SCON_Mean', 'MFCCs13Mean', 'delMFCCs13', 'del2MFCCs13', 'MelSpec','MFCCs20', 'MFCCs1','MFCCs2', 'MFCCs3','MFCCs4', 'MFCCs5','MFCCs6', 'MFCCs7','MFCCs8','MFCCs9','MFCCs10', 'MFCCs11','MFCCs12', 'MFCCs13', 'Cry_Reason'])

def remove_noise(audio_file):
    cry_data = lr.load(audio_file, sr = 22050,mono = True)
    cry_data = cry_data[0]     
    reduced_noise = nr.reduce_noise(y = cry_data, sr=22050, thresh_n_mult_nonstationary=2,stationary=False)
    return reduced_noise

def calculate_amplitude_envelope(signal, FRAME_SIZE, HOP_LENGTH):
    """Calculate the amplitude envelope of a signal with a given frame size and hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), HOP_LENGTH): 
        amplitude_envelope_current_frame = max(signal[i:i+FRAME_SIZE]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
        z = np.array(amplitude_envelope) 
    return z

def extract_amplitude_envelope(audio_file, FRAME_SIZE, HOP_LENGTH):
    ae = calculate_amplitude_envelope(audio_file, FRAME_SIZE, HOP_LENGTH)
    ae_array=np.array(ae)
    ae_me=ae_array.mean()
    return ae_me

def extract_rms(audio_file, FRAME_SIZE, HOP_LENGTH):
    rms = lr.feature.rms(y=audio_file, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms_array=np.array(rms)
    rms_me=rms_array.mean()
    return rms_me

def extract_zcr(audio_file, FRAME_SIZE, HOP_LENGTH):
    zcr = lr.feature.zero_crossing_rate(y=audio_file, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    zcr_array=np.array(zcr)
    zcr_me=zcr_array.mean()
    return zcr_me

def extract_stft(audio_file, FRAME_SIZE, HOP_LENGTH):
    stft=lr.stft(audio_file, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
    stft_mag=np.abs(stft) ** 2
    stft_array=np.array(stft_mag)
    stft_me=stft_array.mean()
    return stft_me

def extract_sc(audio_file, FRAME_SIZE, HOP_LENGTH):
    sc = lr.feature.spectral_centroid(y=audio_file, sr=22050, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    sc_array = np.array(sc)
    sc_me=sc_array.mean()
    return sc_me

def extract_sban(audio_file, FRAME_SIZE, HOP_LENGTH):
    sban = lr.feature.spectral_bandwidth(y=audio_file, sr=22050, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    sban_array=np.array(sban)
    sban_me=sban_array.mean()
    return sban_me

def extract_scon(audio_file, FRAME_SIZE, HOP_LENGTH):
    S = np.abs(lr.stft(audio_file))
    scon = lr.feature.spectral_contrast(S=S, sr=22050)
    scon_array=np.array(scon)
    scon_me=scon_array.mean()
    return scon_me

def extract_mfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    mfccs_me=mfccs_array.mean()
    return mfccs_me

def extract_delmfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    delmfccs_array = lr.feature.delta(mfccs_array)
    delmfccs_me=delmfccs_array.mean()
    return delmfccs_me

def extract_del2mfccs13(audio_file):
    mfccs_array = lr.feature.mfcc(y=audio_file, n_mfcc=13, sr=22050)
    del2mfccs_array = lr.feature.delta(mfccs_array,order=2)
    del2mfccs_me=del2mfccs_array.mean()
    return del2mfccs_me

def extract_melspec(audio_file):
    mel_spectrogram = lr.feature.melspectrogram(y=audio_file, sr=22050, n_fft=2048, hop_length=512, n_mels=256)
    log_mel_spectrogram = lr.power_to_db(mel_spectrogram)
    spec_df=np.array(log_mel_spectrogram)
    spec_mean=np.mean(spec_df)
    return spec_mean

def extract_mfccs20(audio_file):
    mfccs20_array = lr.feature.mfcc(y=audio_file, n_mfcc=20, sr=22050)
    mfccs20_me=mfccs20_array.mean()
    return mfccs20_me

def extract_mfccs1_13(audio_file, FRAME_SIZE, HOP_LENGTH):
    file_clean = remove_noise(audio_file)
    mfccs13_array = lr.feature.mfcc(y=file_clean, n_mfcc=13, sr=22050)
    zz = mfccs13_array
    a0 = zz[0].mean()
    a1 = zz[1].mean()
    a2 = zz[2].mean()
    a3 = zz[3].mean()
    a4 = zz[4].mean()
    a5 = zz[5].mean()
    a6 = zz[6].mean()
    a7 = zz[7].mean()
    a8 = zz[8].mean()
    a9 = zz[9].mean()
    a10 = zz[10].mean()
    a11 = zz[11].mean()
    a12 = zz[12].mean()
    mfccs = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]
    return mfccs

def feature_extractor(audio_file, FRAME_SIZE, HOP_LENGTH):
    file_clean = remove_noise(audio_file)
    a = extract_amplitude_envelope(file_clean, FRAME_SIZE, HOP_LENGTH)
    b = extract_rms(file_clean, FRAME_SIZE, HOP_LENGTH)
    c = extract_zcr(file_clean, FRAME_SIZE, HOP_LENGTH)
    d = extract_stft(file_clean, FRAME_SIZE, HOP_LENGTH)
    e = extract_sc(file_clean, FRAME_SIZE, HOP_LENGTH)
    f = extract_sban(file_clean, FRAME_SIZE, HOP_LENGTH)
    g = extract_scon(file_clean, FRAME_SIZE, HOP_LENGTH)
    h = extract_mfccs13(file_clean)
    i = extract_delmfccs13(file_clean)
    j = extract_del2mfccs13(file_clean)
    k = extract_melspec(file_clean)
    l = extract_mfccs20(file_clean)
    df_pred= pd.DataFrame([[a,b,c,d,e,f,g,h,i,j,k,l]], columns = ['Amplitude_Envelope_Mean','RMS_Mean', 'ZCR_Mean', 'STFT_Mean', 'SC_Mean', 'SBAN_Mean', 'SCON_Mean', 'MFCCs13Mean', 'delMFCCs13', 'del2MFCCs13', 'MelSpec','MFCCs20'])
    return df_pred

option = st.selectbox('How would you like to use this application?', ('Try out the default file', 'Upload your own audio file', 'Record your own audio'))
if option == 'Try out the default file':
    x = '/Users/mac/Desktop/Smart_Cradle/donateacry_corpus_cleaned_and_updated_data/tired/06c4cfa2-7fa6-4fda-91a1-ea186a4acc64-1430029246453-1.7-f-26-ti.wav'
    fe = feature_extractor(x, FRAME_SIZE, HOP_LENGTH)
    fe_more = extract_mfccs1_13(x, FRAME_SIZE, HOP_LENGTH)
    fe['MFCCs1'] = fe_more[0]
    fe['MFCCs2'] = fe_more[1]
    fe['MFCCs3'] = fe_more[2]
    fe['MFCCs4'] = fe_more[3]
    fe['MFCCs5'] = fe_more[4]
    fe['MFCCs6'] = fe_more[5]
    fe['MFCCs7'] = fe_more[6]
    fe['MFCCs8'] = fe_more[7]
    fe['MFCCs9'] = fe_more[8]
    fe['MFCCs10'] = fe_more[9]
    fe['MFCCs11'] = fe_more[10]
    fe['MFCCs12'] = fe_more[11]
    fe['MFCCs13'] = fe_more[12]
    st.audio(x)
    model = joblib.load('bcd_knn_model.pkl')
    prediction = model.predict(fe)
    if(prediction[0]==0):
        st.write('<p class="big-font">Reason: Belly Pain</p>',unsafe_allow_html=True)
    
    elif(prediction[0]==1):
        st.write('<p class="big-font">Reason: Burping</p>',unsafe_allow_html=True)
    
    elif(prediction[0]==2):
        st.write('<p class="big-font">Reason: Discomfort</p>',unsafe_allow_html=True)
    
    elif(prediction[0]==3):
        st.write('<p class="big-font">Reason: Hungry</p>',unsafe_allow_html=True)
        
    elif(prediction[0]==4):
        st.write('<p class="big-font">Reason: Tired</p>',unsafe_allow_html=True)

elif option == 'Upload your own audio file':
    uploaded_file = st.file_uploader("Choose a file", type=["wav"])
    if uploaded_file is not None:
        with NamedTemporaryFile(suffix="wav") as temp:
            temp.write(uploaded_file.getvalue())
            temp.seek(0)
            model = joblib.load('bcd_knn_model.pkl')
            fe = feature_extractor(temp.name, FRAME_SIZE, HOP_LENGTH)
            fe_more = extract_mfccs1_13(temp.name, FRAME_SIZE, HOP_LENGTH)
            fe['MFCCs1'] = fe_more[0]
            fe['MFCCs2'] = fe_more[1]
            fe['MFCCs3'] = fe_more[2]
            fe['MFCCs4'] = fe_more[3]
            fe['MFCCs5'] = fe_more[4]
            fe['MFCCs6'] = fe_more[5]
            fe['MFCCs7'] = fe_more[6]
            fe['MFCCs8'] = fe_more[7]
            fe['MFCCs9'] = fe_more[8]
            fe['MFCCs10'] = fe_more[9]
            fe['MFCCs11'] = fe_more[10]
            fe['MFCCs12'] = fe_more[11]
            fe['MFCCs13'] = fe_more[12]    
            st.audio(temp.name)           
            prediction = model.predict(fe)
            if(prediction[0]==0):
                st.write('<p class="big-font">Reason: Belly Pain</p>',unsafe_allow_html=True)
            
            elif(prediction[0]==1):
                st.write('<p class="big-font">Reason: Burping</p>',unsafe_allow_html=True)
            
            elif(prediction[0]==2):
                st.write('<p class="big-font">Reason: Discomfort</p>',unsafe_allow_html=True)
            
            elif(prediction[0]==3):
                st.write('<p class="big-font">Reason: Hungry</p>',unsafe_allow_html=True)
                
            elif(prediction[0]==4):
                st.write('<p class="big-font">Reason: Tired</p>',unsafe_allow_html=True)

elif option == 'Record your own audio':
    val = st_audiorec()
    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            wav_bytes = stream.read()
            if wav_bytes is not None:
                with NamedTemporaryFile(suffix="wav") as temp:
                    temp.write(wav_bytes)
                    temp.seek(0)
                    model = joblib.load('bcd_knn_model.pkl')
                    fe = feature_extractor(temp.name, FRAME_SIZE, HOP_LENGTH)
                    fe_more = extract_mfccs1_13(temp.name, FRAME_SIZE, HOP_LENGTH)
                    fe['MFCCs1'] = fe_more[0]
                    fe['MFCCs2'] = fe_more[1]
                    fe['MFCCs3'] = fe_more[2]
                    fe['MFCCs4'] = fe_more[3]
                    fe['MFCCs5'] = fe_more[4]
                    fe['MFCCs6'] = fe_more[5]
                    fe['MFCCs7'] = fe_more[6]
                    fe['MFCCs8'] = fe_more[7]
                    fe['MFCCs9'] = fe_more[8]
                    fe['MFCCs10'] = fe_more[9]
                    fe['MFCCs11'] = fe_more[10]
                    fe['MFCCs12'] = fe_more[11]
                    fe['MFCCs13'] = fe_more[12]                              
                    prediction = model.predict(fe)
                    if(prediction[0]==0):
                        st.write('<p class="big-font">Reason: Belly Pain/Gassy/Needs to poop</p>',unsafe_allow_html=True)
                    
                    if(prediction[0]==1):
                        st.write('<p class="big-font">Reason: Burping</p>',unsafe_allow_html=True)            
                    
                    elif(prediction[0]==2):
                        st.write('<p class="big-font">Reason: Discomfort/Hot/Cold/Wet</p>',unsafe_allow_html=True)
                    
                    elif(prediction[0]==3):
                        st.write('<p class="big-font">Reason: Hungry</p>',unsafe_allow_html=True)
                    
                    elif(prediction[0]==4):
                        st.write('<p class="big-font">Reason: Tired/Sleepy</p>',unsafe_allow_html=True)

                    st.write('Did we get it right? What was it?')
                    col1, col2, col3, col4, col5 = st.columns([2,2,2.3,2,2])
                    with col1:
                        if st.button('BELLY PAIN'):
                            reason = 0
                            fe['Cry_reason'] = reason
                    with col2:
                        if st.button('BURPING'):
                            reason = 1
                            fe['Cry_reason'] = reason
                    with col3:
                        if st.button('DISCOMFORT'):
                            reason = 2
                            fe['Cry_reason'] = reason
                    with col4:
                        if st.button('HUNGRY'):
                            reason = 3
                            fe['Cry_reason'] = reason
                    with col5:
                        if st.button('TIRED'):
                            reason = 4
                            fe['Cry_reason'] = reason                       
