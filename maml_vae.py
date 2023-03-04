import tensorflow as tf
from tensorflow.keras import layers
from mlgm.model import Vae
from mlgm.logger import Logger
from mlgm.algo import Maml_train, Maml_test
from mlgm.sampler import MnistMetaSampler_train, MnistMetaSampler_test
import argparse
parser = argparse.ArgumentParser(description='MAML')
parser.add_argument('--num', help='몇번째 데이터 셋 실험인지', type = int)
parser.add_argument('--batch_size', help='테스트 데이터 개수', type = int)
parser.add_argument('--digits', help='테스트 실험에 사용할 데이터를 제외한 숫자들', nargs="+", type=int)
parser.add_argument('--path', help='체크포인트 경로', type = str)
args = parser.parse_args()
"""
metasampler = MnistMetaSampler_train(
        num = args.num,
        batch_size = args.batch_size,      
        meta_batch_size = 12,  
        train_digits = args.digits,      
        test_digits = list(range(2)))

with tf.Session() as sess:
        model, logger = Vae(
        encoder_layers=[
                layers.Dense(6, activation="relu"),
                layers.Dense(4, activation="relu"),
                layers.Dense(units=(2 * 2))], 
        decoder_layers=[
                layers.Dense(4, activation="relu"),
                layers.Dense(6, activation="relu"),
                layers.Dense(8, activation="relu")], sess=sess), Logger(args.num)  
        
        maml = Maml_train(model, metasampler, sess, logger, num_updates = 5, update_lr = 0.005, meta_lr = 0.001, outliers_fraction = 0.5)
        maml.train(train_itr = 90000)
        """
metasampler = MnistMetaSampler_test(
        num = args.num,
        batch_size = args.batch_size,      
        meta_batch_size = 2, 
        train_digits = args.digits,      
        test_digits = list(range(2)))
    
with tf.Session() as sess:
        model = Vae(
        encoder_layers=[
                layers.Dense(6, activation="relu"),
                layers.Dense(4, activation="relu"),
                layers.Dense(units=(2 * 2))], 
        decoder_layers=[
                layers.Dense(4, activation="relu"),
                layers.Dense(6, activation="relu"),
                layers.Dense(8, activation="relu")], sess=sess)

        maml = Maml_test(model, metasampler, sess, num_updates = 1, update_lr = 0.005, meta_lr = 0, outliers_fraction = 0.01)
        maml.test(test_itr = 1, restore_model_path = args.path)
        