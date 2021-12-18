
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar los modelos
model_esp = load_model('modelos/gru_model_lyrics_esp.h5')
model_eng = load_model('modelos/gru_model_lyrics_eng.h5')


# Vocabulario y modelo segun el idioma
def idioma(idioma):
    if idioma == 'esp':
        vocab = ['\n', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\x92', '\x93', '\x94', '¡', '¿', 'à', 'á', 'é', 'í', 'ñ', 'ó', 'ú', 'ü', 'е', '\u2005', '—', '‘', '‚', '…', '\u205f']
        model = model_esp

    elif idioma == 'eng':
        vocab = ['\n', ' ', '!', '"', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        model = model_eng

    return vocab, model


# Funcion para generar el texto
def generar_texto(model, num_generate, temperature, start_string, char2idx, idx2char):
    input_eval = [char2idx[s] for s in start_string]   # de string a numero (vectorizacion)
    input_eval = tf.expand_dims(input_eval, 0)   # dimensionsion
    texto_generado = []   # array vacio para guardar los resultados
    model.reset_states()   # limpiar los estados ocultos de la RNN

    for i in range(num_generate):   
        predicciones = model(input_eval)   # prediccion para un solo caracter
        predicciones = tf.squeeze(predicciones, 0)  # remover el batch

        # usar una distribucion categorica para predecir el caracter devuelto por el modelo
        # una temperatura mas alta aumenta la probabilidad de seleccionar un caracter menos probable
        # mas bajo --> mas predecible
        predicciones = predicciones / temperature
        prediccion_id = tf.random.categorical(predicciones, num_samples=1)[-1,0].numpy()

        # El carácter predicho como la siguiente entrada al modelo
        # junto con el estado oculto anterior
        # Entonces el modelo hace la próxima predicción basada en el caracter anterior
        input_eval = tf.expand_dims([prediccion_id], 0)
        # Desvectorizar el numero y agregar al texto generado
        texto_generado.append(idx2char[prediccion_id])

    return (start_string + ''.join(texto_generado))




def main():
    # Titulo general
    st.title('Generador de Letras de Canciones')
    # Titulo de la barra lateral
    st.sidebar.header('Ingresa los Parametros')

    # Parametros de la barra lateral
    idioma = ["Español", "Ingles"]
    selec_idioma = st.sidebar.selectbox("Idioma de la letra", idioma)
    num_caracteres = st.sidebar.slider('Numero de Caracteres de la Letra', 50, 1000, 500)
    palabra_inicial = st.sidebar.text_input("Palabra inicial")

    # Boton para generar el texto con los parametros elegidos
    if st.button('Generar'):
        if selec_idioma == 'Español':
            vocab, model = idioma('esp')

        elif selec_idioma == 'Ingles':
            vocab, model = idioma('eng')

        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = vocab

        texto_generado = generar_texto(
                        model,
                        num_generate=num_caracteres,
                        temperature=1,
                        start_string=palabra_inicial,
                        char2idx=char2idx,
                        idx2char=idx2char)
    
        st.text(texto_generado)




if __name__ == '__main__':
    main()

