from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, concatenate

def create_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Branch
    cnn = Conv1D(64, 3, activation='relu')(inputs)
    cnn = MaxPooling1D(2)(cnn)
    
    # LSTM Branch
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = LSTM(64)(lstm)
    
    combined = concatenate([cnn, lstm])
    output = Dense(1, activation='sigmoid')(combined)
    
    return Model(inputs=inputs, outputs=output)
