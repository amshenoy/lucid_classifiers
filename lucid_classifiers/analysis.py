import pickle
import os
import tensorflow as tf
import numpy as np
from lucid_utils.classification.lucid_algorithm import classify as lucid_classify
from lucid_utils.classification.lucid_algorithm_data import classify as metric_classify
##returns [self.num_pixels, self.density, self.radius, self.curvature_radius, self.line_residual, self.circle_residual, self.width, self.avg_neighbours]

from collections import Counter

def neural_predict(cluster):
    tf.reset_default_graph()

    classes = ["gamma", "beta", "muon", "proton", "alpha", "others"]
    this_dir, this_filename = os.path.split(__file__)
    model_name = os.path.sep + 'model5.meta'
    LOG_DIR = os.path.join(this_dir, "neural_model")

    checkpoint = tf.train.latest_checkpoint(LOG_DIR)
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.import_meta_graph(LOG_DIR+model_name)
        saver.restore(sess, checkpoint)
        X = tf.get_collection("X")[0]
        predict_op = tf.get_collection('predict_op')[0]
        full_op = tf.get_collection('full_op')[0]

        index = sess.run(predict_op, feed_dict={X: [metric_classify(cluster)]})[0]
        return classes[index]

def classify(cluster, tp=''):
    metric_int = [ int(x) for x in metric_classify(cluster) ]
    this_dir, this_filename = os.path.split(__file__)
    LOG_DIR = os.path.join(this_dir, "benchmark_classifiers")
    classes = ["gamma","beta","muon","proton","alpha","others"]
    
    if tp == "neural":
        out = neural_predict(cluster)
    elif tp in ["svm","knn","dt","rf"]:
        with open(os.path.join(LOG_DIR, tp + '.pkl'), 'rb') as f:
           clf = pickle.load(f, encoding='latin1')
        out = classes[int(clf.predict([metric_int])/2)]
    elif tp == "lucid":
        out = lucid_classify(cluster)
    else:
        preds = []
        preds.append(neural_predict(cluster))
        for tp in ["svm","knn","dt","rf"]:
            with open(os.path.join(LOG_DIR, tp + '.pkl'), 'rb') as f:
               clf = pickle.load(f, encoding='latin1')
            preds.append(classes[int(clf.predict([metric_int])/2)])
        preds.append(lucid_classify(cluster))
        #print(preds)
        out = Counter(preds).most_common(1)[0][0]
    return out
