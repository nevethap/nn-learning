from src import input as ip
from keras import backend as K

import numpy as np
import sklearn.metrics as sm


def evaluate(X_test, Y_test, vgg16_model, top_model):

    test_data = []
    test_data.extend(X_test[0:10])
    test_data.extend(X_test[100:110])
    test_data = np.asarray(test_data)
    print(len(test_data))
    test_labels = np.hstack((Y_test[0:10], Y_test[100:110]))
    print(test_labels)

    print('Generating bottleneck features . . . ')

    bottleneck_features = vgg16_model.predict(test_data)

    print('Done!')

    predictions = top_model.predict_classes(bottleneck_features)

    print('Actual')
    print(test_labels)
    print('Predictions')
    print(np.concatenate(predictions))
    print('Recall : ' + str(sm.recall_score(test_labels, predictions)))
    print('Precision : ' + str(sm.precision_score(test_labels, predictions)))

    return {'predictions' : predictions, 'actuals' : test_labels,
            'recall' : sm.recall_score(test_labels, predictions),
            'precision' : sm.precision_score(test_labels, predictions)}


def run_evaluation():
    X_test, Y_test = ip.input_images()
    vgg16_model = ip.load_vgg16_model()
    top_model = ip.load_top_model()
    evaluate(X_test, Y_test, vgg16_model, top_model)


if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    run_evaluation()
