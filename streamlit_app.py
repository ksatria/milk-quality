import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X_train = pd.read_hdf('milk_quality_classification.h5', key='fitur_training')
y_train = pd.read_hdf('milk_quality_classification.h5', key='label_training')

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

st.title("Milk Quality")
st.write(
    "Isikan data-data yang dibutuhkan berikut:"
)

pH = st.number_input('Nilai pH:', min_value=3.0, max_value=9.5, value=3.0, step=0.1)
temp = st.number_input('Nilai suhu:', min_value=34, max_value=90, value=34, step=1)
taste = st.radio('Rasa:', [0, 1], captions=['Buruk', 'Baik'])
odor = st.radio('Bau', [0, 1], captions=['Busuk', 'Sedap'])
fat = st.radio('Lemak', [0, 1], captions=['Rendah', 'Tinggi'])
turb = st.radio('Kekeruhan', [0, 1], captions=['Rendah', 'Tinggi'])
colour = st.number_input('Nilai warna:', min_value=240, max_value=255, value=240, step=1)

if st.button('Tentukan Grade', type='primary'):
    test_data = np.array([float(pH),float(temp),float(taste),float(odor),float(fat),float(turb),float(colour)])
    test_data = np.reshape(test_data,(1,-1))
    
    hasil = knn.predict(test_data)

    st.write('Prediksi grade :')
    st.text(hasil[0])