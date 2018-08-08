import tensorflow as tf


class Prepare(object):
    # def read_records(self, tasknameList, max_len=25, epochs=1, batch_size=128):
    #     all_train = []
    #     for i in range(len(tasknameList)):
    #         taskname = tasknameList[i]
    #         train_queue = tf.train.string_input_producer([taskname], shuffle=False, num_epochs=1)
    #         reader = tf.TFRecordReader()
    #         _, serialized_example = reader.read(train_queue)
    #         features = tf.parse_single_example(
    #             serialized_example,
    #             features={
    #                 'label': tf.VarLenFeature(tf.int64),
    #                 'query1': tf.VarLenFeature(tf.int64),
    #                 'query2': tf.VarLenFeature(tf.int64)
    #
    #             })
    #
    #         label = tf.sparse_tensor_to_dense(features['label'])
    #         query1 = tf.sparse_tensor_to_dense(features['query1'])
    #         query2 = tf.sparse_tensor_to_dense(features['query2'])
    #
    #         label = tf.cast(label, tf.int32)
    #         query1 = tf.cast(query1, tf.int32)
    #         query2 = tf.cast(query2, tf.int32)
    #
    #         label = tf.reshape(label, [2])
    #         query1 = tf.reshape(query1, [max_len])
    #         query2 = tf.reshape(query2, [max_len])
    #
    #         if epochs > 1:
    #             print ("tf.train.shuffle_batch")
    #             label_batch, query1_batch_serialized, query2_batch_serialized = tf.train.shuffle_batch(
    #                 [label, query1, query2],
    #                 batch_size=batch_size,
    #                 num_threads=2,
    #                 capacity=10000 + 3 * batch_size,
    #                 min_after_dequeue=10000)
    #
    #         else:
    #             print ("tf.train.batch")
    #             label_batch, query1_batch_serialized, query2_batch_serialized = tf.train.batch(
    #                 [label, query1, query2],
    #                 batch_size=batch_size,
    #                 num_threads=2,
    #                 capacity=10000 + 3 * batch_size)
    #
    #         all_train.append([i, label_batch, query1_batch_serialized, query2_batch_serialized])
    #     return all_train

    def read_records(self, taskname, max_len=25, epochs=1, batch_size=128):
        train_queue = tf.train.string_input_producer([taskname], shuffle=False, num_epochs=epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(train_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.VarLenFeature(tf.int64),
                'query1': tf.VarLenFeature(tf.int64),
                'query2': tf.VarLenFeature(tf.int64)

            })

        label = tf.sparse_tensor_to_dense(features['label'])
        query1 = tf.sparse_tensor_to_dense(features['query1'])
        query2 = tf.sparse_tensor_to_dense(features['query2'])

        label = tf.cast(label, tf.int32)
        query1 = tf.cast(query1, tf.int32)
        query2 = tf.cast(query2, tf.int32)

        label = tf.reshape(label, [2])
        query1 = tf.reshape(query1, [max_len])
        query2 = tf.reshape(query2, [max_len])

        if epochs > 1:
            print ("tf.train.shuffle_batch")
            label_batch, query1_batch_serialized, query2_batch_serialized = tf.train.shuffle_batch(
                [label, query1, query2],
                batch_size=batch_size,
                num_threads=4,
                capacity=10000 + 5 * batch_size,
                min_after_dequeue=10000)

        else:
            print ("tf.train.batch")
            label_batch, query1_batch_serialized, query2_batch_serialized = tf.train.batch(
                [label, query1, query2],
                batch_size=batch_size,
                num_threads=2,
                capacity=10000 + 3 * batch_size)

        return [label_batch, query1_batch_serialized, query2_batch_serialized]

    def processTFrecords(self, wordVocab, savePath, max_len=25, taskNumber=2):
        def pad_sentence(sentence, sequence_length=25, padding_word="<UNK/>"):
            if len(sentence) < sequence_length:
                num_padding = sequence_length - len(sentence)
                new_sentence = sentence + [padding_word] * num_padding
            else:
                new_sentence = sentence[:sequence_length]
            return new_sentence

        total_lines = []
        for i in range(taskNumber):  ## represent task i
            train_lines = 0
            filename = savePath + "train-" + str(i) + ".tfrecords"
            writer = tf.python_io.TFRecordWriter(filename)
            openFileName = savePath + "train." + str(i)
            for line in open(openFileName):
                line = line.strip().strip("\n").split("\t")
                if len(line) != 3: continue

                label = [0, 1] if line[0] == "1" else [1, 0]
                query1 = line[1]
                query1 = pad_sentence(query1.split(" "), sequence_length=max_len)
                query1 = wordVocab.to_index_sequenceList(query1)
                if len(query1) != max_len: continue

                query2 = line[2]
                query2 = pad_sentence(query2.split(" "), sequence_length=max_len)
                query2 = wordVocab.to_index_sequenceList(query2)
                if len(query2) != max_len: continue

                train_lines += 1
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=label)),
                        'query1': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=query1)),
                        'query2': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=query2))
                    }))
                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()
            total_lines.append(train_lines)
        return total_lines

    def processTFrecords_test(self, wordVocab, savePath, test_path, max_len=25, taskNumber=1):
        def pad_sentence(sentence, sequence_length=25, padding_word="<UNK/>"):
            if len(sentence) < sequence_length:
                num_padding = sequence_length - len(sentence)
                new_sentence = sentence + [padding_word] * num_padding
            else:
                new_sentence = sentence[:sequence_length]
            return new_sentence

        totalLines = 0
        for i in range(taskNumber):
            filename = savePath + "test-" + str(i) + ".tfrecords"
            writer = tf.python_io.TFRecordWriter(filename)
            openFileName = test_path
            for line in open(openFileName):
                line = line.strip().strip("\n").split("\t")
                if len(line) != 3: continue

                label = [0, 1] if line[0] == "1" else [1, 0]
                query1 = line[1]
                query1 = pad_sentence(query1.split(" "), sequence_length=max_len)
                query1 = wordVocab.to_index_sequenceList(query1)
                if len(query1) != max_len: continue

                query2 = line[2]
                query2 = pad_sentence(query2.split(" "), sequence_length=max_len)
                query2 = wordVocab.to_index_sequenceList(query2)
                if len(query2) != max_len: continue

                totalLines += 1
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        'label': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=label)),
                        'query1': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=query1)),
                        'query2': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=query2))
                    }))
                serialized = example.SerializeToString()
                writer.write(serialized)
            writer.close()
        return totalLines

    def processTFrecords_hasDone(self, savePath, taskNumber=2):
        total_lines = []
        for i in range(taskNumber):  ## represent task i
            train_lines = 0
            openFileName = savePath + "train." + str(i)
            for line in open(openFileName):
                line = line.strip().strip("\n").split("\t")
                if len(line) != 3: continue
                train_lines += 1

            total_lines.append(train_lines)
        return total_lines

    def processTFrecords_test_hasDone(self, test_path, taskNumber=1):
        totalLines = 0
        for i in range(taskNumber):
            openFileName = test_path
            for line in open(openFileName):
                line = line.strip().strip("\n").split("\t")
                if len(line) != 3: continue
                totalLines += 1

        return totalLines
