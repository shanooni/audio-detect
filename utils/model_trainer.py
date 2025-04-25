import numpy as np


# def ensemble_predict(X_test, cnn_input, ml_models, cnn_model, weights):
#     ml_probs = [model.predict_proba(X_test) for model in ml_models.values()]
#     cnn_probs = cnn_model.predict(cnn_input)
    
#     final_probs = (
#         weights['svm'] * ml_probs[0] +
#         weights['rf'] * ml_probs[1] +
#         weights['dt'] * ml_probs[2] +
#         weights['cnn'] * cnn_probs
#     )
    
#     return np.argmax(final_probs, axis=1)

def ensemble_predict(X_test, X_test_cnn, ml_models, cnn_model, weights):
    ml_probs = [model.predict_proba(X_test) for model in ml_models.values()]
    cnn_probs = cnn_model.predict(X_test_cnn)
    
    final_probs = (
        weights['svm'] * ml_probs[0] +
        weights['rf'] * ml_probs[1] +
        weights['dt'] * ml_probs[2] +
        weights['cnn'] * cnn_probs
    )
    
    return np.argmax(final_probs, axis=1)
