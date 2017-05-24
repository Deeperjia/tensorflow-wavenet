Speech-to-Text-WaveNet : End-to-end sentence level Chinese speech recognition using DeepMind's WaveNet
=
A tensorflow implementation for Chinese speech recognition based on DeepMind's WaveNet: A Generative Model for Raw Audio. ([Hereafter the Paper]( https://arxiv.org/abs/1609.03499))

Version
---
Current Version : 0.0.1

Dependencies
---
1. python == 3.5
2. tensorflow == 1.0.0
3. librosa == 0.5.0

Dataset
---
[清华30小时中文数据集](http://data.cslt.org/thchs30/standalone.html)

Directories
---
1. cache: save data featrue and word dictionary
2. data: wav files and related labels
3. model: save the models

Network model
---
1. Data random shuffle per epoch
2. Xavier initialization
3. Adam optimization algorithms
4. Batch Normalization

Train the network
---
python3 train.py

Test the network
---
python3 test.py

Other resources
---
1. [TensorFlow练习15: 中文语音识别](http://blog.topspeedsnail.com/archives/10696#more-10696)
2. [ibab's WaveNet(speech synthesis) tensorflow implementationt](https://github.com/ibab/tensorflow-wavenet)
3. [buriburisuri's WaveNet(English speech recognition) tensorflow and sugartensor implementationt](https://github.com/buriburisuri/speech-to-text-wavenet#version)
