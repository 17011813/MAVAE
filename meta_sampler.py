import tensorflow as tf
class MetaSampler:
    def __init__(self, batch_size, meta_batch_size):  
        self._batch_size = batch_size * 2              
        self._meta_batch_size = meta_batch_size
        self._distribution = None
        self._train_iterator = None
        self._test_iterator = None
        self._ids_per_label = {}        
        self._train_dataset = self._gen_dataset()  

    @property
    def meta_batch_size(self):
        return self._meta_batch_size
    def restart_train_dataset(self, sess):
        sess.run(self._train_iterator.initializer)

    def _gen_dataset(self, test=False):
        raise NotImplementedError
    def build_inputs_and_labels(self):
        raise NotImplementedError

    def init_iterators(self, sess):  
        sess.run(self._train_iterator.initializer)
        train_handle = sess.run(self._train_iterator.string_handle())

        return train_handle                            

    def _gen_metadata(self, handle):
        self._train_iterator = self._train_dataset.make_initializable_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, self._train_dataset.output_types, self._train_dataset.output_shapes)
        meta_batch_sym = iterator.get_next()   
        all_input_batches, all_label_batches = [], []

        for i in range(self._meta_batch_size):
            batch_input_sym = meta_batch_sym[0][i * self._batch_size: (i + 1) * self._batch_size]  
            batch_label_sym = meta_batch_sym[1][i * self._batch_size: (i + 1) * self._batch_size]
            all_input_batches.append(batch_input_sym)
            all_label_batches.append(batch_label_sym)

        return tf.stack(all_input_batches), tf.stack(all_label_batches) 