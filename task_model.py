## encoding=utf8
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs


class Base_model(object):
    def __init__(self,
                 max_len=25,
                 vocab_size=1,
                 embedding_size=1,
                 filter_sizes=1,
                 num_filters=1,
                 num_hidden=1,
                 fix_word_vec=True,
                 word_vocab=None,
                 l2_reg_lambda=0.0,
                 adv=True,
                 diff=True,
                 sharedTag=True):
        self.input_left = tf.placeholder(tf.int32, [None, max_len], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len], name="input_right")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.input_task = tf.placeholder(tf.int32, name="input_task")
        print("input_left", self.input_left.name)
        print("input_right", self.input_right.name)
        print("dropout_keep_prob", self.dropout_keep_prob.name)
        print ("input_y", self.input_y.name)
        print ("input_task", self.input_task.name)

        self.one_hot_input_task = tf.reshape(tf.one_hot(self.input_task, 2, axis=-1),
                                             shape=[1, 2])
        medium = tf.slice(self.input_y, begin=[0, 0], size=[-1, 1])
        self.one_hot_input_task = tf.matmul(tf.fill(tf.shape(medium), 1.0),
                                            self.one_hot_input_task)

        self.sharedTag = sharedTag
        self.diff = diff
        self.adv = adv
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.word_vocab = word_vocab
        self.fix_word_vec = fix_word_vec
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.max_len = max_len
        self.num_hidden = num_hidden
        self.l2_reg_lambda = l2_reg_lambda

        self.embedded_chars_left, self.embedded_chars_right = self.lookupLayer(self.fix_word_vec,
                                                                               self.word_vocab,
                                                                               self.vocab_size)

    def func_shared(self):
        self.shared_out = self.extractLayer()
        print("base_model shared_out: ", self.shared_out.name)

    def func_adv(self):
        if self.adv:
            print ("has loss_adv")
            self.loss_adv, self.l2_loss_adv = self.adversarial_loss()
        else:
            print ("no loss_adv")
            self.loss_adv, self.l2_loss_adv = 0, 0

    def adversarial_loss(self):
        shared_out = flip_gradient(self.shared_out)
        specfic_feature, l2_loss_hidden = self.hidden_layer(x=shared_out,
                                                            input_size=2 * self.num_filters_total + 1,
                                                            output_size=self.num_hidden)

        specfic_feature = tf.nn.dropout(specfic_feature, self.dropout_keep_prob, name="dropout")
        logits, l2_loss = self.linearLayer(specfic_feature,
                                           self.num_hidden,
                                           2)

        loss_adv = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_input_task, logits=logits))

        return loss_adv, l2_loss + l2_loss_hidden

    def hidden_layer(self, x, input_size, output_size):
        with tf.variable_scope(name_or_scope='hidden_layer_shared', reuse=None):
            W = tf.get_variable(name="hidden_layer_W",
                                initializer=tf.contrib.layers.xavier_initializer(),
                                shape=[input_size, output_size])
            print(W.name)
            b = tf.get_variable(initializer=tf.constant(0.1, shape=[output_size]),
                                name="hidden_layer_b")
            l2_loss_hidden = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            hidden_output = self.leaky_relu(tf.nn.xw_plus_b(x, W, b, name="hidden_layer_output"))
            return hidden_output, l2_loss_hidden

    def linearLayer(self, input, input_size, output_size):
        with tf.variable_scope(name_or_scope='linear_Layer_shared', reuse=None):
            W = tf.get_variable("linear_Layer_W",
                                shape=[input_size, output_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            # W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1),
            #                 name="W_linear_0")
            b = tf.get_variable(initializer=tf.constant(0.1, shape=[output_size]),
                                name="linear_Layer_b")
            out_put = tf.nn.xw_plus_b(input, W, b, name="linear_Layer_output")
            l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            print(W.name)
            return out_put, l2_loss

    def extractLayer(self):
        pooled_outputs_left, pooled_outputs_right = self.convLayer()
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool_left = tf.reshape(tf.concat(axis=3, values=pooled_outputs_left), [-1, self.num_filters_total],
                                 name='h_pool_left')
        h_pool_right = tf.reshape(tf.concat(axis=3, values=pooled_outputs_right), [-1, self.num_filters_total],
                                  name='h_pool_right')

        with tf.name_scope("similarity"):
            W = tf.get_variable("W_similarity_base",
                                shape=[self.num_filters_total, self.num_filters_total],
                                initializer=tf.contrib.layers.xavier_initializer())
            transform_left = tf.matmul(h_pool_left, W)
            sims = tf.reduce_sum(tf.multiply(transform_left, h_pool_right), 1, keep_dims=True)
            print(sims.name)

        extract_output = tf.concat(axis=1, values=[h_pool_left, sims, h_pool_right], name='new_input')
        return extract_output

    def convLayer(self):
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                            name="W-%s" % filter_size)
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b-%s" % filter_size)
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                conv = tf.nn.conv2d(self.embedded_chars_left, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))  # conv: [batch_size, 20-2+1, 1, out_channels]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                conv = tf.nn.conv2d(self.embedded_chars_right, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs_right.append(pooled)
        print(W.name)
        return pooled_outputs_left, pooled_outputs_right

    def lookupLayer(self, fix_word_vec, word_vocab, vocab_size):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if fix_word_vec:
                word_vec_trainable = False
                wordInitial = tf.constant(word_vocab.word_vecs)
                W = tf.get_variable("word_embedding",
                                    trainable=word_vec_trainable,
                                    initializer=wordInitial,
                                    dtype=tf.float32)
                print("fix_word_vec")
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0), name="W_Embedding")

            # [batch_size, max_length, embedding_size, 1]
            embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_left), -1)
            embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_right), -1)
        return embedded_chars_left, embedded_chars_right

    def leaky_relu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


class MTLModel_0(object):
    def __init__(self, objects):
        self.objects = objects
        with tf.name_scope("MTLModel_0"):
            self.loss_diff, self.general_loss, l2_loss, l2_loss_hidden = self.general_func()

            self.l2_loss = l2_loss_hidden + l2_loss
            self.total_loss = self.general_loss + 0.01 * self.loss_diff + self.objects.l2_reg_lambda * self.l2_loss

    def general_func(self):
        general_out = self.extractLayer()
        if self.objects.sharedTag:
            print("--with shared layer--")
            general_feature = tf.concat([general_out, self.objects.shared_out], axis=1)
            general_feature, l2_loss_hidden = self.hidden_layer(x=general_feature,
                                                                input_size=4 * self.num_filters_total + 2,
                                                                output_size=self.objects.num_hidden)
        else:
            print("--without shared layer--")
            general_feature = general_out
            general_feature, l2_loss_hidden = self.hidden_layer(x=general_feature,
                                                                input_size=2 * self.num_filters_total + 1,
                                                                output_size=self.objects.num_hidden)

        general_feature = tf.nn.dropout(general_feature, self.objects.dropout_keep_prob, name="dropout")
        general_feature_linear, l2_loss = self.linearLayer(general_feature,
                                                           self.objects.num_hidden,
                                                           2)
        self.general_prob = general_prob = tf.nn.softmax(general_feature_linear, name='prob')
        general_acc = self.get_acc(general_prob)
        general_losses = self.general_loss(logits=general_prob,
                                           labels=self.objects.input_y)
        general_loss = tf.reduce_mean(general_losses)
        self.acc = general_acc
        if self.objects.diff:
            print ("has loss_diff")
            loss_diff_0 = self.diff_loss(self.objects.shared_out, general_out)
        else:
            print ("no loss_diff")
            loss_diff_0 = 0
        return loss_diff_0, general_loss, l2_loss, l2_loss_hidden

    def hidden_layer(self, x, input_size, output_size):
        W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1),
                        name="W_hidden_layer_0")
        # W = tf.get_variable(name="hidden_layer_0",
        #                     initializer=tf.contrib.layers.xavier_initializer(),
        #                     shape=[input_size, output_size])
        print(W.name)
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b_layer_0")
        l2_loss_hidden = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        hidden_output = self.leaky_relu(tf.nn.xw_plus_b(x, W, b, name="hidden_output_layer_0"))
        return hidden_output, l2_loss_hidden

    def diff_loss(self, shared_feat, task_feat):
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

    def get_acc(self, prob):
        correct_predictions = tf.equal(tf.argmax(prob, 1),
                                       tf.argmax(self.objects.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy

    def linearLayer(self, input, input_size, output_size):
        # W = tf.get_variable("W_linear_0",
        #                     shape=[input_size, output_size],
        #                     initializer=tf.contrib.layers.xavier_initializer())
        W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1),
                        name="W_linear_0")
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b_linear_0")
        out_put = tf.nn.xw_plus_b(input, W, b, name="output_linear_0")
        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        print(W.name)
        return out_put, l2_loss

    def leaky_relu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

    def general_loss(self, logits, labels):
        scores = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        labels2 = tf.matmul(labels, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))

        tmp = tf.multiply(tf.subtract(1.0, labels2), tf.square(scores))
        temp2 = tf.multiply(labels2, tf.square(tf.maximum(tf.subtract(1.0, scores), 0.0)))
        sum = tf.add(tmp, temp2)
        return sum

    def extractLayer(self):
        pooled_outputs_left, pooled_outputs_right = self.convLayer()
        self.num_filters_total = self.objects.num_filters * len(self.objects.filter_sizes)
        h_pool_left = tf.reshape(tf.concat(axis=3, values=pooled_outputs_left), [-1, self.num_filters_total],
                                 name='h_pool_left')
        h_pool_right = tf.reshape(tf.concat(axis=3, values=pooled_outputs_right), [-1, self.num_filters_total],
                                  name='h_pool_right')

        with tf.name_scope("similarity"):
            W = tf.get_variable("W_similarity_0",
                                shape=[self.num_filters_total, self.num_filters_total],
                                initializer=tf.contrib.layers.xavier_initializer())
            transform_left = tf.matmul(h_pool_left, W)
            sims = tf.reduce_sum(tf.multiply(transform_left, h_pool_right), 1, keep_dims=True)
            print(sims.name)

        extract_output = tf.concat(axis=1, values=[h_pool_left, sims, h_pool_right], name='new_input')
        return extract_output

    def convLayer(self):
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(self.objects.filter_sizes):
            filter_shape = [filter_size, self.objects.embedding_size, 1, self.objects.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                            name="W-%s" % filter_size)
            b = tf.Variable(tf.constant(0.1, shape=[self.objects.num_filters]), name="b-%s" % filter_size)
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                conv = tf.nn.conv2d(self.objects.embedded_chars_left, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))  # conv: [batch_size, 25-2+1, 1, out_channels]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.objects.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                conv = tf.nn.conv2d(self.objects.embedded_chars_right, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.objects.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs_right.append(pooled)
        print(W.name)
        return pooled_outputs_left, pooled_outputs_right


class MTLModel_1(object):
    def __init__(self, objects):
        self.objects = objects
        with tf.name_scope("MTLModel_1"):
            self.loss_diff, self.specfic_loss, l2_loss, l2_loss_hidden = self.specfic_func()

            self.l2_loss = l2_loss_hidden + l2_loss
            self.total_loss = self.specfic_loss + 0.01 *self.loss_diff + self.objects.l2_reg_lambda * self.l2_loss

    def specfic_func(self):
        specfic_out = self.extractLayer()
        if self.objects.sharedTag:
            print("--with shared layer--")
            specfic_feature = tf.concat([specfic_out, self.objects.shared_out], axis=1)
            specfic_feature, l2_loss_hidden = self.hidden_layer(x=specfic_feature,
                                                                input_size=4 * self.num_filters_total + 2,
                                                                output_size=self.objects.num_hidden)
        else:
            print("--without shared layer--")
            specfic_feature = specfic_out
            specfic_feature, l2_loss_hidden = self.hidden_layer(x=specfic_feature,
                                                                input_size=2 * self.num_filters_total + 1,
                                                                output_size=self.objects.num_hidden)

        specfic_feature = tf.nn.dropout(specfic_feature, self.objects.dropout_keep_prob, name="dropout")
        specfic_feature_linear, l2_loss = self.linearLayer(specfic_feature,
                                                           self.objects.num_hidden,
                                                           2)
        self.specfic_prob = specfic_prob = tf.nn.softmax(specfic_feature_linear, name='prob')
        print("specfic_prob: ", specfic_prob.name)
        specfic_acc = self.get_acc(specfic_prob)
        specfic_losses = self.general_loss(logits=specfic_prob,
                                           labels=self.objects.input_y)
        specfic_loss = tf.reduce_mean(specfic_losses)
        if self.objects.diff:
            print ("has loss_diff")
            loss_diff_1 = self.diff_loss(self.objects.shared_out, specfic_out)
        else:
            print ("no loss_diff")
            loss_diff_1 = 0
        self.acc = specfic_acc
        return loss_diff_1, specfic_loss, l2_loss, l2_loss_hidden

    def hidden_layer(self, x, input_size, output_size):
        W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1),
                        name="W_hidden_layer_1")
        # W = tf.get_variable(name="hidden_layer_1",
        #                     initializer=tf.contrib.layers.xavier_initializer(),
        #                     shape=[input_size, output_size])
        print(W.name)
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b_layer_1")
        l2_loss_hidden = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        hidden_output = self.leaky_relu(tf.nn.xw_plus_b(x, W, b, name="hidden_output_layer_1"))
        return hidden_output, l2_loss_hidden

    def diff_loss(self, shared_feat, task_feat):
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

    def get_acc(self, prob):
        correct_predictions = tf.equal(tf.argmax(prob, 1),
                                       tf.argmax(self.objects.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy

    def linearLayer(self, input, input_size, output_size):
        W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1),
                        name="W_linear_1")
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b_linear")
        out_put = tf.nn.xw_plus_b(input, W, b, name="output_linear")
        l2_loss = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        print(W.name)
        return out_put, l2_loss

    def leaky_relu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

    def general_loss(self, logits, labels):
        scores = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        labels2 = tf.matmul(labels, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))

        tmp = tf.multiply(tf.subtract(1.0, labels2), tf.square(scores))
        temp2 = tf.multiply(labels2, tf.square(tf.maximum(tf.subtract(1.0, scores), 0.0)))
        sum = tf.add(tmp, temp2)
        return sum

    def extractLayer(self):
        pooled_outputs_left, pooled_outputs_right = self.convLayer()
        self.num_filters_total = self.objects.num_filters * len(self.objects.filter_sizes)
        h_pool_left = tf.reshape(tf.concat(axis=3, values=pooled_outputs_left), [-1, self.num_filters_total],
                                 name='h_pool_left')
        h_pool_right = tf.reshape(tf.concat(axis=3, values=pooled_outputs_right), [-1, self.num_filters_total],
                                  name='h_pool_right')

        with tf.name_scope("similarity"):
            W = tf.get_variable("W_similarity_1",
                                shape=[self.num_filters_total, self.num_filters_total],
                                initializer=tf.contrib.layers.xavier_initializer())
            transform_left = tf.matmul(h_pool_left, W)
            sims = tf.reduce_sum(tf.multiply(transform_left, h_pool_right), 1, keep_dims=True)
            print(sims.name)

        extract_output = tf.concat(axis=1, values=[h_pool_left, sims, h_pool_right], name='new_input')
        return extract_output

    def convLayer(self):
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(self.objects.filter_sizes):
            filter_shape = [filter_size, self.objects.embedding_size, 1, self.objects.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                            name="W-%s" % filter_size)
            b = tf.Variable(tf.constant(0.1, shape=[self.objects.num_filters]), name="b-%s" % filter_size)
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                conv = tf.nn.conv2d(self.objects.embedded_chars_left, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))  # conv: [batch_size, 20-2+1, 1, out_channels]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.objects.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                conv = tf.nn.conv2d(self.objects.embedded_chars_right, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                h = self.leaky_relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.objects.max_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs_right.append(pooled)
        print(W.name)
        return pooled_outputs_left, pooled_outputs_right


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
