# encoding=utf8
import tensorflow as tf
from base_model import *

TASK_NUM = 2


class MTLModel():
    def __init__(self,
                 max_len=25,
                 filter_sizes=1,
                 num_filters=1,
                 num_hidden=1,
                 word_vocab=None,
                 l2_reg_lambda=0.0,
                 learning_rate=1,
                 adv=True):
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_left_0 = tf.placeholder(tf.int32, [None, max_len], name="input_left_0")
        self.input_right_0 = tf.placeholder(tf.int32, [None, max_len], name="input_right_0")
        self.input_y_0 = tf.placeholder(tf.int32, [None, 2], name="input_y_0")
        self.input_task_0 = tf.placeholder(tf.int32, name="input_task_0")
        print("input_left", self.input_left_0.name)
        print("input_right", self.input_right_0.name)
        print("dropout_keep_prob", self.dropout_keep_prob.name)
        print ("input_y", self.input_y_0.name)
        print ("input_task", self.input_task_0.name)
        self.input_left_1 = tf.placeholder(tf.int32, [None, max_len], name="input_left_1")
        self.input_right_1 = tf.placeholder(tf.int32, [None, max_len], name="input_right_1")
        self.input_y_1 = tf.placeholder(tf.int32, [None, 2], name="input_y_1")
        self.input_task_1 = tf.placeholder(tf.int32, name="input_task_1")

        self.adv = adv
        self.word_vocab = word_vocab
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.max_len = max_len
        self.num_hidden = num_hidden
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate

        wordInitial = tf.constant(word_vocab.word_vecs)
        self.word_embed = tf.get_variable("word_embedding",
                                          trainable=False,
                                          initializer=wordInitial,
                                          dtype=tf.float32)

        self.shared_conv = ConvLayer(layer_name='conv_shared',
                                     filter_sizes=self.filter_sizes,
                                     num_filters=self.num_filters)
        self.shared_linear = LinearLayer('linear_shared', TASK_NUM, True)
        self.tensors = []

        with tf.name_scope("task_0"):
            self.build_task_graph(task_label=self.input_task_0,
                                  labels=self.input_y_0,
                                  sentence_0=self.input_left_0,
                                  sentence_1=self.input_right_0)

        with tf.name_scope("task_1"):
            self.build_task_graph(task_label=self.input_task_1,
                                  labels=self.input_y_1,
                                  sentence_0=self.input_left_1,
                                  sentence_1=self.input_right_1)

    def build_task_graph(self,
                         task_label,
                         labels,
                         sentence_0,
                         sentence_1):
        sentence_0 = tf.nn.embedding_lookup(self.word_embed, sentence_0)
        sentence_1 = tf.nn.embedding_lookup(self.word_embed, sentence_1)
        sentence_0 = tf.nn.dropout(sentence_0, self.dropout_keep_prob)
        sentence_1 = tf.nn.dropout(sentence_1, self.dropout_keep_prob)
        ######## layer
        conv_layer = ConvLayer(layer_name='conv_task',
                               filter_sizes=self.filter_sizes,
                               num_filters=self.num_filters)
        ########
        conv_out_0 = conv_layer(sentence_0)
        conv_out_0 = max_pool(conv_outs=conv_out_0,
                              max_len=self.max_len,
                              num_filters=self.num_filters)
        conv_out_1 = conv_layer(sentence_1)
        conv_out_1 = max_pool(conv_outs=conv_out_1,
                              max_len=self.max_len,
                              num_filters=self.num_filters)
        task_output = tf.concat(axis=1, values=[conv_out_0, conv_out_1], name='task_output')

        shared_out_0 = self.shared_conv(sentence_0)
        shared_out_0 = max_pool(conv_outs=shared_out_0,
                                max_len=self.max_len,
                                num_filters=self.num_filters)
        shared_out_1 = self.shared_conv(sentence_1)
        shared_out_1 = max_pool(conv_outs=shared_out_1,
                                max_len=self.max_len,
                                num_filters=self.num_filters)
        shared_output = tf.concat(axis=1, values=[shared_out_0, shared_out_1], name='shared_output')

        if self.adv:
            feature = tf.concat([task_output, shared_output], axis=1)
        else:
            feature = task_output

        feature = tf.nn.dropout(feature, self.dropout_keep_prob)

        # Map the features to 2 classes
        linear = LinearLayer('linear', 2, True)
        logits, loss_l2 = linear(feature)

        logits_prob = tf.nn.softmax(logits, name='prob')
        print ("logits_prob: ",logits_prob.name)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
        loss_ce = tf.reduce_mean(xentropy)

        loss_adv, loss_adv_l2 = self.adversarial_loss(shared_output, task_label, labels)
        loss_diff = self.diff_loss(shared_output, task_output)

        if self.adv:
            print ("Adv is True")
            loss = loss_ce + 0.05 * loss_adv + self.l2_reg_lambda * (loss_l2 + loss_adv_l2) + 0.01 * loss_diff
        else:
            print ("Adv is False")
            loss = loss_ce + self.l2_reg_lambda * loss_l2

        pred = tf.argmax(logits, axis=1)
        labels_2 = tf.argmax(labels, axis=1)
        acc = tf.cast(tf.equal(pred, labels_2), tf.float32)
        acc = tf.reduce_mean(acc)

        self.tensors.append((acc, loss, loss_adv, loss_ce))

    def adversarial_loss(self, feature, task_label, y_label):
        '''make the task classifier cannot reliably predict the task based on
        the shared feature
        '''
        # input = tf.stop_gradient(input)
        feature = flip_gradient(feature)  ## let adv_loss increasing
        feature = tf.nn.dropout(feature, self.dropout_keep_prob)

        # Map the features to TASK_NUM classes
        logits, loss_l2 = self.shared_linear(feature)

        label = tf.reshape(tf.one_hot(task_label, 2, axis=-1),
                           shape=[1, 2])
        medium = tf.slice(y_label, begin=[0, 0], size=[-1, 1])
        label = tf.matmul(tf.fill(tf.shape(medium), 1.0), label)

        loss_adv = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

        return loss_adv, loss_l2

    def diff_loss(self, shared_feat, task_feat):
        '''Orthogonality Constraints from https://github.com/tensorflow/models,
        in directory research/domain_adaptation
        '''
        task_feat -= tf.reduce_mean(task_feat, 0)  # 按列求得平均
        shared_feat -= tf.reduce_mean(shared_feat, 0)

        task_feat = tf.nn.l2_normalize(task_feat, 1)  # 按行归一化
        shared_feat = tf.nn.l2_normalize(shared_feat, 1)

        correlation_matrix = tf.matmul(task_feat, shared_feat, transpose_a=True)

        cost = tf.reduce_mean(tf.square(correlation_matrix)) * 0.01

        cost = tf.where(cost > 0, cost, 0, name='value')
        assert_op = tf.Assert(tf.is_finite(cost), [cost])
        with tf.control_dependencies([assert_op]):
            loss_diff = tf.identity(cost)

        return loss_diff

    # def build_train_op(self):
    #     self.train_ops = []
    #     for _, loss, _ in self.tensors:
    #         global_step = tf.Variable(0, name="global_step", trainable=False)
    #         train_op = optimize(loss,
    #                             global_step,
    #                             self.learning_rate)
    #         self.train_ops.append([train_op, global_step])

    def build_train_op(self):
        self.train_ops = []
        for _, loss, _, _ in self.tensors:
            train_op = optimize(loss, self.learning_rate)
            self.train_ops.append(train_op)
