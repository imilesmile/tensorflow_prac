import collections
import tensorflow as tf

class BatchedInput(collections.namedtuple(
    "BatchedInput", ("initializer", "source", "target", "source_len"))):
  pass

def get_iterator(raw_data, batch_size, num_steps):

  output_buffer_size = batch_size * 100
  source_data = raw_data.shuffle(output_buffer_size, reshuffle_each_iteration=True)
  source_data = source_data.map(
    lambda src: tf.string_split([src], ',').values, num_parallel_calls=4).prefetch(output_buffer_size)
  source_data = source_data.map(
    lambda src: tf.string_to_number(src, tf.int32), num_parallel_calls=4).prefetch(output_buffer_size)
  source_data = source_data.map(
    lambda src: src[:num_steps+1], num_parallel_calls=4).prefetch(output_buffer_size)
  source_target_data = source_data.map(
    lambda src: (src[:tf.size(src)-1], src[1:], tf.size(src)-1)).prefetch(output_buffer_size)
  batch_data = source_target_data.padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([num_steps]), tf.TensorShape([num_steps]), tf.TensorShape([])),
    padding_values=(0, 0, 0)).filter(lambda x,y,z: tf.equal(tf.shape(x)[0], batch_size))
  batched_iter = batch_data.make_initializable_iterator()
  (source, target, source_len) = batched_iter.get_next()

  return BatchedInput(initializer=batched_iter.initializer,
                      source=source, target=target, source_len=source_len)
