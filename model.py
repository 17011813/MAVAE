import tensorflow as tf
class Model:
    def __init__(self,
                 layers,
                 sess,
                 learning_rate=0.001,
                 model_name="model"):
        self._sess = sess
        self._x = None
        self._y = None
        self._out = None
        self._param_in = None
        self._param_out = None
        self._layers = layers
        self._loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._optimize = None
        self._name = model_name
        self._name_scope = None

    def build(self):
        self._x = tf.placeholder(**self._param_in)
        self._y = tf.placeholder(**self._param_out)     
        self._out = self.build_forward_pass(self._x)   
        loss_sym = self.build_loss(self._y, self._out) 
        gradients_sym, weights_sym = self.build_gradients(loss_sym)
        self.build_apply_gradients(gradients_sym, weights_sym)

    def get_variables(self):
        return tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name_scope )

    def _set_tensors(self, layer_in, layer, use_tensors):
        supported_types = [ tf.keras.layers.Dense, tf.keras.layers.Conv2D ]  
        if type(layer) in supported_types:
            if (layer.kernel.name in use_tensors) and (layer.bias.name in use_tensors):
                kernel_var = layer.kernel
                bias_var = layer.bias
                layer.kernel = use_tensors[layer.kernel.name]
                layer.bias = use_tensors[layer.bias.name]
                layer_out = layer(layer_in)
                layer.kernel = kernel_var
                layer.bias = bias_var
            else:
                layer_out = layer(layer_in)
        else:
            layer_out = layer(layer_in)
        return layer_out

    def build_forward_pass(self, input_tensor, use_tensors=None, name=None):    
        layer_in = input_tensor
        with tf.variable_scope(name, default_name=self._name, values=[layer_in]) as forward_scope:
            if not self._name_scope:
                self._name_scope = forward_scope._name_scope
            for layer in self._layers:
                if use_tensors:
                    layer_out = self._set_tensors(layer_in, layer, use_tensors)
                else:
                    layer_out = layer(layer_in)
                layer_in = layer_out

        return layer_out

    def build_loss(self, label, model_out, name=None):
        if not name:
            name = self._name + "_loss"
        with tf.variable_scope(name, values=[label, model_out]):
            return self._loss_fn(label, model_out)   

    def build_gradients(self, loss_sym, fast_params=None):
        grads, params = {}, {}
        if fast_params:
            for name, w in fast_params.items():
                params.update({name: w})
                grads.update({name: tf.gradients(loss_sym, w)[0]})
        else:
            for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name_scope):
                params.update({param.name: param})
                grads.update({param.name: tf.gradients(loss_sym, param)[0]})
        return grads, params

    def build_apply_gradients(self, gradients_sym, weights_sym):
        grad_var = [(gradients_sym[w_name], weights_sym[w_name]) for w_name in weights_sym]
        self._optimize = self._optimizer.apply_gradients(grad_var)

    def optimize(self, x, y):
        feed_dict = {self._x: x, self._y: y}
        self._sess.run([self._optimize], feed_dict=feed_dict)

    def restore_model(self, save_path):
        var_list = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name_scope)]
        saver = tf.train.Saver(var_list)
        saver.restore(self._sess, save_path)