#-*- coding:utf-8 -*-

from __future__ import print_function
from model import Model
from utils import SpeechLoader

import tensorflow as tf  # 1.0.0
import numpy as np
import librosa
import os
 
# 语音识别
# 把batch_size改为1
def speech_to_text():
    n_mfcc = 60

    # load data
    speech_loader = SpeechLoader()

    # load model
    model = Model(speech_loader.vocab_size, n_mfcc=n_mfcc, is_training=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        for j in range(750,755):
            # extract feature
            wav_file = os.path.join(os.getcwd(),'data','wav','test','D4','D4_'+str(j)+'.wav')
            wav, sr = librosa.load(wav_file, mono=True)
            mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr, n_mfcc=n_mfcc), axis=0), [0,2,1])
            mfcc = mfcc.tolist()

            # fill 0
            while len(mfcc[0]) < speech_loader.wav_max_len:
                mfcc[0].append([0] * n_mfcc)

            # word dict
            wmap = {value:key for key, value in speech_loader.wordmap.items()}

            # recognition
            saver.restore(sess, tf.train.latest_checkpoint('model'))
            decoded = tf.transpose(model.logit, perm=[1, 0, 2])
            decoded, probs = tf.nn.ctc_beam_search_decoder(decoded, model.seq_len, top_paths=1, merge_repeated=True)
            predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values) + 1
            output, probs = sess.run([predict, probs], feed_dict={model.input_data: mfcc})
            
            # print result
            words = ''
            for i in range(len(output[0])):
                words += wmap.get(output[0][i], -1)

            print("---------------------------")
            print("Input: " + wav_file)
            print("Output: " + words)


if __name__ == '__main__':
        speech_to_text()
        
    