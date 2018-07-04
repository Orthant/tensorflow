# An improvement on the textprocessor.py provided by tflearn
# as an example. This adds more summaries to the Tensorboard 
# such as:
#   "Text"      : Copy of the outputted generated text for 
#                 keeping track of the progress of learning.
#
#   "Beholder"  : This feature
#
#   "Projector" : So you can play with the tSNE and such.
#
#   "Checkpoint": So the model reloads and saves regularly

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import urllib
import tflearn
import re
import pickle
import tensorflow as tf

from datetime import datetime as dt
from tflearn.data_utils import textfile_to_semi_redundant_sequences, random_sequence_from_textfile
from tensorboard.plugins.beholder import Beholder
from builtins import any as b_any

# Save the pickle file
char_idx_file = "char_idx.pickle"
# Model name to save to
model_name = "textgenerator"

parser = argparse.ArgumentParser(description='Parse a textfile to generate LSTM output')
parser.add_argument('filename')
parser.add_argument('-t', '--temp', help=
                    'If temp is specified, a value of 0.0 to 2.0 is recommended.' +
                    'A value closer to 0 will result in output closer to the input,'
                    'so a higher value is riskier or more novel. Defaults to 0.8',
                    required=False,
                    nargs=1,
                    type=float)

parser.add_argument('-l', '--length', help=
                    'Optional length of text sequences to analyse, the higher the value ' +
                    'the more memory you will need. Defaults to 25',
                    required=False,
                    default=25,
                    nargs=1,
                    type=int)

args = vars(parser.parse_args())
path = args['filename']

if not os.path.isfile(path):
  print("Couldn't find the file specified. Is the path correct?")
  print("Path used: '" + path + "'")
  sys.exit(1)

# Filename for naming runs 
filename=os.path.basename(path)
# Name to be used in the "Text" tab of Tensorboard
text_summary_name=filename.replace('.txt', '').replace('.\\', '').replace('./', '')
# Run name to be used in Tensorboard
run_name=model_name+'_'+text_summary_name+'_'+dt.now().strftime('%Y%m%d-%H%M%S')

temp=None
if args['temp'] and args['temp'][0] is not None:
  temp = args['temp'][0]
  print('Temperature set to ', temp)
  if temp > 2:
    print("Temperature out of suggested range. Suggested temp range is 0.0-2.0")
  if temp < 0:
    print("Temperature out of suggested range. Setting to default (0.8)")
    temp=0.8
else:
  temp=0.8

if args['length'] and args['length'][0] is not 25:
  maxlen = args['length'][0]
  print('Sequence max length set to ', maxlen)
else:
  maxlen=25
  print('Using default sequence length of 25')

X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

if os.path.isfile(char_idx_file):
  print('Loading previous char_idx_file')
  char_idx=pickle.load(open(char_idx_file, 'rb'))
else:
  print('Dumping char_idx_file')
  pickle.dump(char_idx, open(char_idx_file, 'wb'))

# Because my GPU kept running out of memory (Quadro P600)
tflearn.init_graph(gpu_memory_fraction=0.4, log_device=False)

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')

g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', restore=True)
m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=0.5,
                              tensorboard_verbose=3,
                              tensorboard_dir='./run/',
                              checkpoint_path='./run/' + model_name)

if os.path.exists(model_name+'.index'):
  print('--- LOADING EXISTING MODEL ---')
  m.load(model_name)

with tf.name_scope('generated_text') as scope:
  valid_placeholder=tf.placeholder(tf.string, name='generated_text')
  text_summary=tf.summary.text(text_summary_name, valid_placeholder)
  summary_op=tf.summary.merge([text_summary])

beholder = Beholder('./run/')

for i in range(5):
  seed = random_sequence_from_textfile(path, maxlen)
  print('-- STARTING RUN NUMBER %s --' & i)
  m.fit(X, Y, validation_set=0.2, batch_size=64, n_epoch=1, run_id=run_name, snapshot_epoch=True)

  print('-- TESTING WITH TEMPERATURE OF %s --' % temp)
  gentext = m.generate(6000, temperature=temp, seq_seed=seed)
  print('-- GENERATION COMPLETED --')

  # Add it to the summary placeholder
  _sess = m.session
  _graph=_sess.graph
  _logdir=m.trainer.summ_writer.get_logdir()
  _step = int(m.trainer.global_step.eval(session=_sess))
  _writer = tf.summary.FileWriter(_logdir, graph=_graph)
  output_summary = _sess.run(summary_op, feed_dict={
    valid_placeholder: [gentext]
  })
  _writer.add_summary(output_summary, global_step=_step)
  _writer.flush()
  _writer.close()
  m.trainer.saver.save(_sess, './run/' + model_name + '.ckpt', _step)
  beholder.update(_sess)

m.save(model_name)