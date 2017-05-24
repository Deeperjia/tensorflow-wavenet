#-*- coding:utf-8 -*-
__author__ = 'Deeper'
import tensorflow as tf 
import numpy as np 
import os
import codecs
import librosa
from six.moves import cPickle, reduce, map
from collections import Counter
import random

class SpeechLoader():

	def __init__(self, wav_path=None, label_file=None, batch_size=1, n_mfcc=20, encoding='utf-8'):
		self.batch_size = batch_size
		self.encoding = encoding
		self.n_mfcc = n_mfcc

		# path setting
		data_dir = os.path.join(os.getcwd(), 'cache', 'mfcc'+str(n_mfcc))
		# data cache
		wavs_file = os.path.join(data_dir, "wavs.file")
		vocab_file = os.path.join(data_dir,"vocab.file")
		mfcc_tensor = os.path.join(data_dir, "mfcc.tensor")
		label_tensor = os.path.join(data_dir, "label.tensor")

		# data process
		if not (os.path.exists(vocab_file) and os.path.exists(mfcc_tensor) and os.path.exists(label_tensor)):
			print("reading wav files")
			self.preprocess(wav_path, label_file, wavs_file, vocab_file, mfcc_tensor, label_tensor)
		else:
			print("loading preprocessed files")
			self.load_preprocessed(vocab_file, mfcc_tensor, label_tensor)

		# minibatch
		self.create_batches()
		# pointer reset
		self.reset_batch_pointer()

	def preprocess(self, wav_path, label_file, wavs_file, vocab_file, mfcc_tensor, label_tensor):
		def handle_file(dirpath, filename):
			if filename.endswith('.wav') or filename.endswith('.WAV'):
				filename_path = os.path.join(dirpath, filename)
				if os.stat(filename_path).st_size < 240000:
					return
				return filename_path

		# read label file
		labels_dict = {}
		with codecs.open(label_file,"r", encoding=self.encoding) as f:
			for label in f:
				label = label.strip('\n')
				labels_id = label.split(' ',1)[0]
				labels_text = label.split(' ',1)[1]
				labels_dict[labels_id] = labels_text
		# print("",len(labels_dict)) # 10000
		
		# wav files
		wav_files = []
		if wav_path:
			for (dirpath, dirnames, filenames) in os.walk(wav_path):
				for filename in filenames:
					if handle_file(dirpath,filename):
						wav_files.append(handle_file(dirpath,filename))
		print("初始样本数：", len(wav_files)) #样本数

		# data filter and feature extraction
		wav_files_filter = []
		labels_filter = []
		self.mfcc_tensor = []
		self.wav_max_len = 0
		cnt = 0
		for wav_file in wav_files:
			wav_id = os.path.basename(wav_file).split('.')[0]
			if wav_id in labels_dict:
				print('样本'+str(cnt), wav_file)
				labels_filter.append(labels_dict[wav_id])
				wav_files_filter.append(wav_file)
				# mfcc feature
				wav_file, sr = librosa.load(wav_file, mono=True)
				mfcc = np.transpose(librosa.feature.mfcc(wav_file, sr, n_mfcc=self.n_mfcc),[1,0])
				self.mfcc_tensor.append(mfcc.tolist())
				cnt += 1
		self.wav_max_len = max(len(mfcc) for mfcc in self.mfcc_tensor)
		print("样本总数：", cnt)
		print("最长的语音：", self.wav_max_len)
		# print(len(wav_files_filter), len(labels_filter),len(wav2mfcc)) # assert check dimensions

		with open(wavs_file, 'wb') as f:
			cPickle.dump(wav_files_filter, f)

		with open(mfcc_tensor, 'wb') as f:
			cPickle.dump(self.mfcc_tensor, f)

		# vocab file
		vocabs = []
		for label in labels_filter:
			vocabs += [word for word in label]
		count = Counter(vocabs)
		count_pairs = sorted(count.items(), key=lambda x:-x[1])
		words, _ = zip(*count_pairs)
		self.wordmap = dict(zip(words, range(len(words))))

		self.vocab_size = len(words)
		print("词汇表大小：",len(words))

		with open(vocab_file,'wb') as f:
			cPickle.dump(self.wordmap, f)

		# label vector
		label_encoder = lambda word: self.wordmap.get(word, len(words))
		self.label_tensor = [list(map(label_encoder, label)) for label in labels_filter]
		self.label_max_len = max(len(label) for label in self.label_tensor)
		print("最长的句子：", self.label_max_len)

		with open(label_tensor,'wb') as f:
			cPickle.dump(self.label_tensor, f)

	def load_preprocessed(self, vocab_file, mfcc_tensor, label_tensor):
		with open(vocab_file, 'rb') as f:
			self.wordmap = cPickle.load(f) 
		self.vocab_size = len(self.wordmap)
		print("词汇表大小：",self.vocab_size)

		with open(mfcc_tensor, 'rb') as f:
			self.mfcc_tensor = cPickle.load(f)
		self.wav_max_len = max(len(mfcc) for mfcc in self.mfcc_tensor)
		print("最长的语音：", self.wav_max_len)

		with open(label_tensor, 'rb') as f:
			self.label_tensor = cPickle.load(f)
		self.label_max_len = max(len(label) for label in self.label_tensor)
		print("最长的句子：", self.label_max_len)

	def create_batches(self):
		self.n_batches = len(self.mfcc_tensor) // self.batch_size
		if self.n_batches==0:
			assert False, "Not enough data. Make seq_length and batch_size small."
		
		self.mfcc_tensor = self.mfcc_tensor[:self.n_batches*self.batch_size]
		self.label_tensor = self.label_tensor[:self.n_batches*self.batch_size]

		# random shuffle the data
		if len(self.mfcc_tensor) != len(self.label_tensor):
			assert False, "Data length does not match the label length!"

		data_tensor = []
		for i in range(len(self.mfcc_tensor)):
			data_tensor.append([self.mfcc_tensor[i], self.label_tensor[i]])

		random.shuffle(data_tensor)
		self.mfcc_tensor = []
		self.label_tensor = []
		for i in range(len(data_tensor)):
			self.mfcc_tensor.append(data_tensor[i][0])
			self.label_tensor.append(data_tensor[i][1])

		# create batches
		self.x_batches = []
		self.y_batches = []

		for i in range(self.n_batches):
			from_index = i*self.batch_size
			to_index = from_index + self.batch_size
			mfcc_batches = self.mfcc_tensor[from_index:to_index]
			label_batches = self.label_tensor[from_index:to_index]
			# 补零对齐
			for mfcc in mfcc_batches:
				while len(mfcc) < self.wav_max_len:
					mfcc.append([0]*self.n_mfcc)
			for label in label_batches:
				while len(label) < self.label_max_len:
					label.append(0)

			self.x_batches.append(mfcc_batches)
			self.y_batches.append(label_batches)

	def next_batch(self):
		x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
		self.pointer += 1
		return x, y

	def reset_batch_pointer(self):
		self.pointer = 0