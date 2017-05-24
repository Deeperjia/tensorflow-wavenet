#-*- coding:utf-8 -*-

from __future__ import print_function
from utils import SpeechLoader
from model import Model
import tensorflow as tf #1.0.0
import time
import os

def train():
	# setting parameters
	batch_size = 32
	n_epoch = 100
	n_mfcc = 60

	# load speech data
	wav_path = os.path.join(os.getcwd(),'data','wav','train')
	label_file = os.path.join(os.getcwd(),'data','doc','trans','train.word.txt')
	speech_loader = SpeechLoader(wav_path, label_file, batch_size, n_mfcc)
	n_out = speech_loader.vocab_size

	# load model
	model = Model(n_out, batch_size=batch_size, n_mfcc=n_mfcc)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		saver = tf.train.Saver(tf.global_variables())

		for epoch in range(n_epoch):
			speech_loader.create_batches() # random shuffle data
			speech_loader.reset_batch_pointer()
			for batch in range(speech_loader.n_batches):
				start = time.time()
				batches_wav, batches_label = speech_loader.next_batch()
				feed = {model.input_data: batches_wav, model.targets: batches_label}
				train_loss, _ = sess.run([model.cost, model.optimizer_op], feed_dict=feed)
				end = time.time()
				print("epoch: %d/%d, batch: %d/%d, loss: %s, time: %.3f."%(epoch, n_epoch, batch, speech_loader.n_batches, train_loss, end-start))

			# save models
			if epoch % 5 ==0:
				saver.save(sess, os.path.join(os.getcwd(), 'model','speech.module'), global_step=epoch)


if __name__ == '__main__':
	train()