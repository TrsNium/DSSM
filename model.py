import tensorflow as tf
import numpy as np

class DSSM():
    def __init__(self, input_dim, l1_dim , l2_dim, l3_dim, NEG, BS):
        input_query = tf.placeholder(dtype=tf.float32, shape=[BS, input_d], name='input_query')
        input_document = tf.placeholder(dtype=tf.float32, shape=[BS, input_d], name='input_document') 

        l1_w = tf.Variable(tf.truncated_normal([input_dim, l1_dim], stddev=0.1), name='l1_w')
        l1_b = tf.Variable(tf.truncated_normal([l1_dim], stddev=0.1), name='l1_b')

        l1_query_out = tf.nn.xw_plus_b(input_query, l1_w, l1_b)
        l1_query_out = tf.nn.relu(l1_query_out, name='l1_querry_out')

        l1_document_out = tf.nn.xw_plus_b(input_document, l1_b, l1_b)
        l1_document_out = tf.nn.relu(l1_document_out, name='l1_document_out')

        l2_w = tf.Variable(tf.truncated_normal([l1_dim, l2_dim], stddev=0.1), name='l2_w')
        l2_b = tf.Variable(tf.truncated_normal([l2_dim], stddev=0.1), name='l2_b')

        l2_query_out = tf.nn.xw_plus_b(l1_query_out, l2_w, l2_b)
        l2_query_out = tf.nn.relu(l2_query_out, name='l2_query_out')

        l2_document_out = tf.nn.xw_plus_b(l1_document_out, l2_w, l2_b)
        l2_document_out = tf.nn.relu(l2_document_out, name='l2_document_out')

        l3_w = tf.Variable(tf.truncated_normal([l2_dim, l3_dim], stddev=0.1), name='l3_w')
        l3_b = tf.Variable(tf.truncated_normal([l3_dim], stddev=0.1), name='l3_b')

        l3_query_out = tf.nn.xw_plus_b(l2_query_out, l3_w, l3_b)
        l3_query_out = tf.nn.relu(l3_query_out, name='l3_query_out')

        l3_document_out = tf.nn.xw_plus_b(l2_document_out, l3_w, l3_b)
        l3_document_out = tf.nn.relu(l3_document_out, name='l3_document_out')

        #cos_sim
        square_query = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(l3_query_out), axis=1, True)), [NEG+1, 1])
        square_doc = tf.sqrt(tf.reduce_sum(tf.square(l3_document_out), 1, True))

        prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
        norm_prod = tf.mul(query_norm, doc_norm)

        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * Gamma

        prob = tf.nn.softmax((cos_sim))
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        loss = -tf.reduce_sum(tf.log(hit_prob))/ BS

        lr = tf.Variable(0.003, False)
        self.__train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
