import collections
import tensorflow as tf


class BatchedInput(collections.namedtuple(
    "BatchedInput", ("initializer", "source_id", "source", "target", "source_len"))):
    pass


def get_iterator(raw_data, batch_size, num_steps):
    output_buffer_size = batch_size * 100
    source_data = raw_data.shuffle(output_buffer_size, reshuffle_each_iteration=True)
    source_data = source_data.map(
        lambda src: tf.string_split([src], ',').values, num_parallel_calls=4).prefetch(output_buffer_size)
    source_data = source_data.map(
        lambda src: tf.string_to_number(src, tf.int32), num_parallel_calls=4).prefetch(output_buffer_size)
    source_data = source_data.map(
        lambda src: src[:num_steps + 1], num_parallel_calls=4).prefetch(output_buffer_size)
    source_target_data = source_data.map(
        lambda src: (src[:tf.size(src) - 1], src[1:], tf.size(src) - 1)).prefetch(output_buffer_size)
    batch_data = source_target_data.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([num_steps]), tf.TensorShape([num_steps]), tf.TensorShape([])),
        padding_values=(0, 0, 0)).filter(lambda x, y, z: tf.equal(tf.shape(x)[0], batch_size))
    batched_iter = batch_data.make_initializable_iterator()
    (source, target, source_len) = batched_iter.get_next()

    return BatchedInput(initializer=batched_iter.initializer, source_id=None, source=source, target=target,
                        source_len=source_len)


def infer_iterator(raw_data, batch_size, num_steps):
    output_buffer_size = batch_size * 100
    source_data = raw_data.shuffle(output_buffer_size, reshuffle_each_iteration=True)
    source_data = source_data.map(
        lambda src: (tf.string_split([src], '\t').values[0],
                     tf.string_split([src], '\t').values[1]),
        num_parallel_calls=4).prefetch(output_buffer_size)
    source_data = source_data.map(
        lambda x, y: (x, tf.string_split([y], ',').values), num_parallel_calls=4).prefetch(output_buffer_size)
    source_data = source_data.map(
        lambda x, y: (x, tf.string_to_number(y, tf.int32)),
        num_parallel_calls=4).prefetch(output_buffer_size)
    source_data = source_data.map(
        lambda x, y: (x, y[:num_steps + 1]), num_parallel_calls=4).prefetch(output_buffer_size)
    source_target_data = source_data.map(
        lambda x, y: (x, y[:tf.size(y) - 1], y[1:], tf.size(y) - 1), batch_size)
    batch = source_target_data.batch(batch_size).filter(
        lambda x, y, z, s: tf.equal(tf.shape(x)[0], batch_size))
    batched_iter = batch.make_initializable_iterator()
    (source_id, source, target, source_len) = batched_iter.get_next()
    return BatchedInput(initializer=batched_iter.initializer,
                        source_id=source_id, source=source, target=target, source_len=source_len)


if __name__ == '__main__':
    with tf.Session() as sess:
        dataset = tf.data.TextLineDataset("./infer")
        dataset_train = tf.data.TextLineDataset("./train")

        # iterator = dataset.make_one_shot_iterator()
        # textline = iterator.get_next()
        # print (textline.eval())

        raw_data = dataset
        raw_data_train = dataset_train
        batch_size = 12
        num_steps = 3
        output_buffer_size = batch_size * 100
        source_data = raw_data.shuffle(output_buffer_size, reshuffle_each_iteration=True)

        source_data = source_data.map(
            lambda src: (tf.string_split([src], '\t').values[0],
                         tf.string_split([src], '\t').values[1]),
            num_parallel_calls=4).prefetch(output_buffer_size)

        source_data = source_data.map(
            lambda x, y: (x, tf.string_split([y], ',').values), num_parallel_calls=4).prefetch(output_buffer_size)

        source_data = source_data.map(
            lambda x, y: (x, tf.string_to_number(y, tf.int32)),
            num_parallel_calls=4).prefetch(output_buffer_size)
        source_data = source_data.map(
            lambda x, y: (x, y[:num_steps + 1]), num_parallel_calls=4).prefetch(output_buffer_size)

        source_target_data = source_data.map(
            lambda x, y: (x, y[:tf.size(y) - 1], y[1:], tf.size(y) - 1), batch_size)
        batch = source_target_data.batch(batch_size).filter(
            lambda x, y, z, s: tf.equal(tf.shape(x)[0], batch_size))
        batched_iter = batch.make_initializable_iterator()
        (source_id, source, target, source_len) = batched_iter.get_next()
        sess.run(batched_iter.initializer)
        next_element = batched_iter.get_next()
        print (sess.run(next_element))

        # iterator = source_target_data.make_one_shot_iterator()
        # textline = iterator.get_next()
        # print (sess.run(textline))
        # #
        # iterator = source_target_data.make_initializable_iterator()
        # textline = iterator.get_next()
        # print (sess.run(textline))
        # #
        # # batched_iter = batch_data.make_initializable_iterator()
        # # # (source, target, source_len) = batched_iter.get_next()
        # # sess.run(batched_iter.initializer)
        # # next_element = batched_iter.get_next()
        # # print (sess.run(next_element))
        # return source_data
        ############################################
        source_data = raw_data_train.shuffle(output_buffer_size, reshuffle_each_iteration=True)
        source_data = source_data.map(
            lambda src: tf.string_split([src], ',').values, num_parallel_calls=4).prefetch(output_buffer_size)
        source_data = source_data.map(
            lambda src: tf.string_to_number(src, tf.int32), num_parallel_calls=4).prefetch(output_buffer_size)
        source_data = source_data.map(
            lambda src: src[:num_steps + 1], num_parallel_calls=4).prefetch(output_buffer_size)
        source_target_data = source_data.map(
            lambda src: (src[:tf.size(src) - 1], src[1:], tf.size(src) - 1)).prefetch(output_buffer_size)
        batch_data = source_target_data.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([num_steps]), tf.TensorShape([num_steps]), tf.TensorShape([])),
            padding_values=(0, 0, 0)).filter(lambda x, y, z: tf.equal(tf.shape(x)[0], batch_size))
        batched_iter = batch_data.make_initializable_iterator()
        (source, target, source_len) = batched_iter.get_next()
        sess.run(batched_iter.initializer)
        next_element = batched_iter.get_next()
        # print (sess.run(next_element))
