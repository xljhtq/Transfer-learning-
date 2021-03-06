#!/usr/bin/env python
# encoding=utf-8
"""
modify:
1.hasTfrecords
2.adv
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from vocab_utils import Vocab
from gene_tfrecords import Prepare
import mtl_model


def main_func(_):
    print(FLAGS)
    save_path = FLAGS.train_dir + "tfFile/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("1. Loading WordVocab data...")
    wordVocab = Vocab()
    wordVocab.fromText_format3(FLAGS.train_dir, FLAGS.wordvec_path)
    sys.stdout.flush()

    prepare = Prepare()
    if FLAGS.hasTfrecords:
        print("2. Has Tfrecords File---Train---")
        total_lines = prepare.processTFrecords_hasDone(savePath=save_path, taskNumber=FLAGS.taskNumber)
    else:
        print("2. Start generating TFrecords File--train...")
        total_lines = prepare.processTFrecords(wordVocab, savePath=save_path, max_len=FLAGS.max_len,
                                               taskNumber=FLAGS.taskNumber)
    print("totalLines_train_0:", total_lines[0])
    print("totalLines_train_1:", total_lines[1])
    sys.stdout.flush()

    test_path = FLAGS.train_dir + FLAGS.test_path
    if FLAGS.hasTfrecords:
        print("3. Has TFrecords File--test...")
        totalLines_test = prepare.processTFrecords_test_hasDone(test_path=test_path, taskNumber=1)
    else:
        print("3. Start generating TFrecords File--test...")
        totalLines_test = prepare.processTFrecords_test(wordVocab,
                                                        savePath=save_path,
                                                        test_path=test_path,
                                                        max_len=FLAGS.max_len,
                                                        taskNumber=1)
    print("totalLines_test:", totalLines_test)
    sys.stdout.flush()

    print("4. Start loading TFrecords File...")
    taskNameList = []
    for i in range(FLAGS.taskNumber):
        string = FLAGS.train_dir + 'tfFile/train-' + str(i) + '.tfrecords'
        taskNameList.append(string)
    print("taskNameList: ", taskNameList)
    sys.stdout.flush()

    ################
    n = total_lines[0] / total_lines[1] + 1 if \
        total_lines[0] % total_lines[1] != 0 else \
        total_lines[0] / total_lines[1]
    print("n: ", n)
    num_batches_per_epoch_train_0 = int(total_lines[0] / FLAGS.batch_size) + 1 if \
        total_lines[0] % FLAGS.batch_size != 0 else int(
        total_lines[0] / FLAGS.batch_size)
    print("batch_numbers_train_0:", num_batches_per_epoch_train_0)
    batch_size_1 = FLAGS.batch_size / n

    num_batches_per_epoch_test = int(totalLines_test / FLAGS.batch_size) + 1 if \
        totalLines_test % FLAGS.batch_size != 0 else \
        int(totalLines_test / FLAGS.batch_size)
    print("batch_numbers_test:", num_batches_per_epoch_test)

    with tf.Graph().as_default():
        all_test = prepare.read_records(
            taskname=save_path + "test-0.tfrecords",
            max_len=FLAGS.max_len,
            epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size)

        all_train_0 = prepare.read_records(
            taskname=taskNameList[0],
            max_len=FLAGS.max_len,
            epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size)

        all_train_1 = prepare.read_records(
            taskname=taskNameList[1],
            max_len=FLAGS.max_len,
            epochs=FLAGS.num_epochs,
            batch_size=batch_size_1)

        print("Loading Model...")
        sys.stdout.flush()

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("------------train model--------------")
            m_train = mtl_model.MTLModel(max_len=FLAGS.max_len,
                                         filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                                         num_filters=FLAGS.num_filters,
                                         num_hidden=FLAGS.num_hidden,
                                         word_vocab=wordVocab,
                                         l2_reg_lambda=FLAGS.l2_reg_lambda,
                                         learning_rate=FLAGS.learning_rate,
                                         adv=FLAGS.adv)
            m_train.build_train_op()
            print("\n\n")

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            has_pre_trained_model = False
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs"))

            print(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            else:
                print("continue training models")
                ckpt = tf.train.get_checkpoint_state(out_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("-------has_pre_trained_model--------")
                    print(ckpt.model_checkpoint_path)
                    has_pre_trained_model = True
                    sys.stdout.flush()

            checkpoint_prefix = os.path.join(out_dir, "model")
            if has_pre_trained_model:
                print("Restoring model from " + ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("DONE!")
                sys.stdout.flush()

            def dev_whole(num_batches_per_epoch_test):
                accuracies = []
                losses = []

                for j in range(num_batches_per_epoch_test):
                    input_y_test, input_left_test, input_centre_test = sess.run(
                        [all_test[0], all_test[1], all_test[2]])
                    loss, accuracy, loss_adv, loss_ce = sess.run(
                        [m_train.tensors[1][1], m_train.tensors[1][0], m_train.tensors[1][2],
                         m_train.tensors[1][3]],
                        feed_dict={
                            m_train.input_task_0: 0,
                            m_train.input_left_0: input_left_real_0,
                            m_train.input_right_0: input_centre_real_0,
                            m_train.input_y_0: input_y_real_0,
                            m_train.dropout_keep_prob: FLAGS.dropout_keep_prob,
                            m_train.input_task_1: 1,
                            m_train.input_left_1: input_left_test,
                            m_train.input_right_1: input_centre_test,
                            m_train.input_y_1: input_y_test,
                        })
                    losses.append(loss_ce)
                    accuracies.append(accuracy)
                # print("specfic_prob: ", prob_test)
                sys.stdout.flush()
                return np.mean(np.array(losses)), np.mean(np.array(accuracies))

            def overfit(dev_accuracy):
                n = len(dev_accuracy)
                if n < 4:
                    return False
                for i in range(n - 4, n):
                    if dev_accuracy[i] > dev_accuracy[i - 1]:
                        return False
                return True

            dev_accuracy = []
            total_train_loss = []

            train_loss_0 = 0
            train_loss_1 = 0
            loss_task_0 = 0
            loss_task_1 = 0
            adv_0 = 0
            adv_1 = 0
            acc_1 = 0
            count = 0
            try:
                while not coord.should_stop():  ## for each epoch
                    for i in range(num_batches_per_epoch_train_0 * FLAGS.num_epochs):  ## for each batch
                        input_y_real_0, input_left_real_0, input_centre_real_0 = sess.run([all_train_0[0],
                                                                                           all_train_0[1],
                                                                                           all_train_0[2]])
                        input_y_real_1, input_left_real_1, input_centre_real_1 = sess.run([all_train_1[0],
                                                                                           all_train_1[1],
                                                                                           all_train_1[2]])

                        # acc, loss, loss_adv = m_train.tensors[0]
                        # _, current_step_0, loss_0, accuracy_0, loss_adv_0 = sess.run(
                        #     [m_train.train_ops[0][0], m_train.train_ops[0][1],
                        #      m_train.tensors[0][1], m_train.tensors[0][0], m_train.tensors[0][2]],
                        #     feed_dict={
                        #         m_train.input_task_0: 0,
                        #         m_train.input_left_0: input_left_real_0,
                        #         m_train.input_right_0: input_centre_real_0,
                        #         m_train.input_y_0: input_y_real_0,
                        #         m_train.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        #         m_train.input_task_1: 1,
                        #         m_train.input_left_1: input_left_real_1,
                        #         m_train.input_right_1: input_centre_real_1,
                        #         m_train.input_y_1: input_y_real_1,
                        #     })
                        # all_loss_adv += loss_adv_0
                        # train_acc += accuracy_0
                        # train_loss_0 += loss_0
                        # train_loss += loss_0
                        #
                        # _, current_step_1, loss_1, accuracy_1, loss_adv_1 = sess.run(
                        #     [m_train.train_ops[1][0], m_train.train_ops[1][1],
                        #      m_train.tensors[1][1], m_train.tensors[1][0], m_train.tensors[1][2]],
                        #     feed_dict={
                        #         m_train.input_task_0: 0,
                        #         m_train.input_left_0: input_left_real_0,
                        #         m_train.input_right_0: input_centre_real_0,
                        #         m_train.input_y_0: input_y_real_0,
                        #         m_train.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        #         m_train.input_task_1: 1,
                        #         m_train.input_left_1: input_left_real_1,
                        #         m_train.input_right_1: input_centre_real_1,
                        #         m_train.input_y_1: input_y_real_1,
                        #     })
                        _, loss_0, accuracy_0, loss_adv_0, loss_ce_0 = sess.run(
                            [m_train.train_ops[0],
                             m_train.tensors[0][1], m_train.tensors[0][0], m_train.tensors[0][2],
                             m_train.tensors[0][3]],
                            feed_dict={
                                m_train.input_task_0: 0,
                                m_train.input_left_0: input_left_real_0,
                                m_train.input_right_0: input_centre_real_0,
                                m_train.input_y_0: input_y_real_0,
                                m_train.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                m_train.input_task_1: 1,
                                m_train.input_left_1: input_left_real_1,
                                m_train.input_right_1: input_centre_real_1,
                                m_train.input_y_1: input_y_real_1,
                            })
                        train_loss_0 += loss_0
                        loss_task_0 += loss_ce_0
                        adv_0 += loss_adv_0

                        _, loss_1, accuracy_1, loss_adv_1, loss_ce_1 = sess.run(
                            [m_train.train_ops[1],
                             m_train.tensors[1][1], m_train.tensors[1][0], m_train.tensors[1][2],
                             m_train.tensors[1][3]],
                            feed_dict={
                                m_train.input_task_0: 0,
                                m_train.input_left_0: input_left_real_0,
                                m_train.input_right_0: input_centre_real_0,
                                m_train.input_y_0: input_y_real_0,
                                m_train.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                m_train.input_task_1: 1,
                                m_train.input_left_1: input_left_real_1,
                                m_train.input_right_1: input_centre_real_1,
                                m_train.input_y_1: input_y_real_1,
                            })
                        train_loss_1 += loss_1
                        loss_task_1 += loss_ce_1
                        adv_1 += loss_adv_1
                        acc_1 += accuracy_1

                        count += 1
                        if count % 500 == 0:
                            print("loss {}, acc {}".format(loss_0, accuracy_0))
                            print("--loss {}, acc {}, loss_adv {}, loss_ce {}--".format(loss_1, accuracy_1, loss_adv_1,
                                                                                        loss_ce_1))
                            sys.stdout.flush()

                        if count % num_batches_per_epoch_train_0 == 0 or \
                                count == num_batches_per_epoch_train_0 * FLAGS.num_epochs:

                            print("train_0: ", count / num_batches_per_epoch_train_0,
                                  " epoch, train_loss_0:", train_loss_0,
                                  "loss_task_0: ", loss_task_0,
                                  "adv_0: ", adv_0)

                            print(
                                "train_1: ", count / num_batches_per_epoch_train_0,
                                " epoch, train_loss_1: ", train_loss_1,
                                "loss_task_1: ", loss_task_1,
                                "adv_1: ", adv_1,
                                "acc_1 : ", acc_1 / num_batches_per_epoch_train_0)

                            total_train_loss.append(loss_task_1)
                            train_loss_0 = 0
                            train_loss_1 = 0
                            loss_task_0 = 0
                            loss_task_1 = 0
                            adv_0 = 0
                            adv_1 = 0
                            acc_1 = 0
                            sys.stdout.flush()

                            print("\n------------------Evaluation:-----------------------")
                            _, accuracy = dev_whole(num_batches_per_epoch_test)
                            dev_accuracy.append(accuracy)
                            print("--------Recently dev accuracy:--------")
                            print(dev_accuracy[-10:])

                            print("--------Recently loss_task_1:------")
                            print(total_train_loss[-10:])
                            if overfit(dev_accuracy):
                                print('-----Overfit!!----')
                                break
                            print("")
                            sys.stdout.flush()

                            # continue
                            path = saver.save(sess, checkpoint_prefix, global_step=count)

                            print("-------------------Saved model checkpoint to {}--------------------".format(path))
                            sys.stdout.flush()
                            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                            output_node_names=[
                                                                                                'task_1/prob'])
                            for node in output_graph_def.node:
                                if node.op == 'RefSwitch':
                                    node.op = 'Switch'
                                    for index in xrange(len(node.input)):
                                        if 'moving_' in node.input[index]:
                                            node.input[index] = node.input[index] + '/read'
                                elif node.op == 'AssignSub':
                                    node.op = 'Sub'
                                    if 'use_locking' in node.attr:
                                        del node.attr['use_locking']

                            with tf.gfile.GFile(FLAGS.train_dir + "runs/mtlmodel_specfic.pb", "wb") as f:
                                f.write(output_graph_def.SerializeToString())
                            print("%d ops in the final graph.\n" % len(output_graph_def.node))




            except tf.errors.OutOfRangeError:
                print("Done")
            finally:
                print("--------------------------finally---------------------------")
                print("current_step:", count)
                coord.request_stop()
                coord.join(threads)

            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordvec_path", default="data/wordvec.vec", help="wordvec_path")
    parser.add_argument("--train_dir", default="./", help="Training dir root")
    parser.add_argument("--test_path", default="tfFile/test.0", help="test path")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size (default: 64)")
    parser.add_argument("--taskNumber", type=int, default=2, help="Number of tfRecordsfile (default: 2)")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--max_len", type=int, default=25, help="max document length of input")
    parser.add_argument("--fix_word_vec", default=True, help="fix_word_vec")
    parser.add_argument("--hasTfrecords", default=True, help="hasTfrecords")
    parser.add_argument("--adv", default=True, help="adv")

    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimensionality of character embedding (default: 64)")
    parser.add_argument("--filter_sizes", default="3,4,5", help="Comma-separated filter sizes (default: '2,3')")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters per filter size (default: 64)")
    parser.add_argument("--num_hidden", type=int, default=100, help="Number of hidden layer units (default: 100)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.0001, help="L2 regularizaion lambda")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="L2 regularizaion lambda")
    parser.add_argument("--allow_soft_placement", default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, help="Log placement of ops on devices")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main_func, argv=[sys.argv[0]] + unparsed)
