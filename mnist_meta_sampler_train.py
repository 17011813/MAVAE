from itertools import permutations
import numpy as np
import pandas as pd
import tensorflow as tf
from mlgm.sampler import MetaSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random, sys

np.set_printoptions(threshold=sys.maxsize)
scaler = MinMaxScaler() 
class MnistMetaSampler_train(MetaSampler):
    def __init__(
            self,
            num,
            batch_size,
            meta_batch_size,
            train_digits,
            test_digits):
        self._train_digits, self._normal_ids_per_label, self._anormal_ids_per_label, self._train_size = list(set(train_digits)), {}, {}, 0 
        df = pd.read_csv('./data/새로운실험/{}/re_{}.csv'.format(num, num))
        self.train_inputs, train_labels, isit = scaler.fit_transform(np.array(df.iloc[:, :-2])), np.array(df['label']).reshape(-1,), np.array(df['anomaly']).reshape(-1,)
        self._plz = isit
        for digit in self._train_digits:                    
            normal_ids = np.where((digit == train_labels) & (isit == 0.0))[0]    
            anomal_ids = np.where((digit == train_labels) & (isit == 1.0))[0]
            self._train_size = self._train_size + len(normal_ids) + len(anomal_ids)  
            random.shuffle(normal_ids)
            random.shuffle(anomal_ids)
            self._normal_ids_per_label.update({digit: normal_ids})  
            self._anormal_ids_per_label.update({digit: anomal_ids})

        super().__init__(batch_size, meta_batch_size)

    def _gen_dataset(self, test=False):
        normal_ids_per_label, anormal_ids_per_label = self._normal_ids_per_label, self._anormal_ids_per_label   
        tasks = []
        while True:
            tasks_remaining = self._meta_batch_size - len(tasks) 
            if tasks_remaining <= 0: break                       
            tasks_to_add = list(permutations(self._train_digits, 1))               
            n_tasks_to_add = min(len(tasks_to_add), tasks_remaining)
            tasks.extend(tasks_to_add[:n_tasks_to_add]) 

        num_inputs_per_meta_batch = (self._batch_size  * self._meta_batch_size) 
        indexs, lbls = np.empty((0, num_inputs_per_meta_batch), dtype=np.int32), np.empty((0, num_inputs_per_meta_batch), dtype=np.int32) 
        data_size = min(self._train_size // num_inputs_per_meta_batch, 1000)   

        for i in range(data_size):
            all_indexs, all_labels = np.array([], dtype=np.int32), np.array([], dtype=np.int32) 
            for task in tasks: 
                labels = np.empty(self._batch_size, dtype=np.int32)
                label_indexs = np.append(np.random.choice(normal_ids_per_label[task[0]], 9), np.random.choice(anormal_ids_per_label[task[0]], 1))   
                label_indexs = np.append(label_indexs, np.random.choice(normal_ids_per_label[task[0]], 9))        
                label_indexs = np.append(label_indexs, np.random.choice(anormal_ids_per_label[task[0]], 1))  
                print("{}에 대한 정상이랑 비정상 잘 뽑혔나~?".format(task), self._plz[label_indexs])
                labels.fill(task[0])                                     
                all_labels, all_indexs = np.append(all_labels, labels), np.append(all_indexs, label_indexs) 
            
            # task [(0), (1)] 각 task 마다 0 라벨에 대한 인덱스들, 1 라벨에 대한 인덱스들이 쫙 순서대로 있고, all_indexs에 순서대로 연결되어있다.
            lbls = np.append(lbls, [all_labels], axis=0)        
            indexs = np.append(indexs, [all_indexs], axis=0)        
        all_indexs_sym = tf.convert_to_tensor(indexs)
        inputs_sym = tf.convert_to_tensor(self.train_inputs, dtype=tf.float32)
        all_inputs = tf.gather(inputs_sym, all_indexs_sym)                    
        all_labels = tf.convert_to_tensor(lbls, dtype=tf.dtypes.int32)       

        return tf.data.Dataset.from_tensor_slices((all_inputs, all_labels)) 
        # 여기서 뱉어내는 test는 tf.gather에서 해당 0 task에 대한 인덱스들(all_inputs)과 그에 상응하는 라벨들(all_labels)

    def build_inputs_and_labels(self, handle):
        slice_size = (self._batch_size // 2) 
        input_batches, label_batches = self._gen_metadata(handle)    
        
        input_a = tf.slice(input_batches, [0, 0, 0], [-1, slice_size, -1])
        input_b = tf.slice(input_batches, [0, slice_size, 0], [-1, -1, -1]) 
        output_a, output_b = input_a, input_b

        real_a, real_b = tf.slice(label_batches, [0, 0], [-1, slice_size]), tf.slice(label_batches, [0, slice_size], [-1, -1])

        print("각 입력과 각 라벨들 : ", input_a.shape, output_a.shape, input_b.shape, output_b.shape, real_a.shape, real_b.shape)
        return input_a, output_a, input_b, output_b, real_a, real_b