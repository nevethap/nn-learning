from src import input as ip
from keras import backend as K

import numpy as np
import sklearn.metrics as sm


def evaluate(X_test, Y_test, vgg16_model, top_model):
    print('Generating bottleneck features . . . ')

    bottleneck_features = vgg16_model.predict(X_test)

    print('Done!')

    predictions = top_model.predict_classes(bottleneck_features)

    print('Actual')
    print(Y_test)
    print('Predictions')
    print(np.concatenate(predictions))
    print('Recall : ' + str(sm.recall_score(Y_test, predictions)))
    print('Precision : ' + str(sm.precision_score(Y_test, predictions)))

    return {'predictions' : predictions, 'actuals' : Y_test,
            'recall' : sm.recall_score(Y_test, predictions),
            'precision' : sm.precision_score(Y_test, predictions)}


def run_evaluation():
    X_test, Y_test = ip.input_images()
    vgg16_model = ip.load_vgg16_model()
    top_model = ip.load_top_model()
    evaluate(X_test, Y_test, vgg16_model, top_model)


if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    run_evaluation()
