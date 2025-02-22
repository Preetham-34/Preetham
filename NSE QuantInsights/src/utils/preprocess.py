from sklearn.preprocessing import MinMaxScaler, RobustScaler

def scale_features(df, method='minmax'):
    """Feature scaling pipeline"""
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = RobustScaler()
        
    return scaler.fit_transform(df), scaler
