#!/bin/bash


source ~/.bashrc

rm -rf /tmp/lvtao/.theano/

# export KMP_AFFINITY=granularity=core,noduplicates,compact,0,0
# export OMP_NUM_THREADS=56

python th_rnn_inference_cpu_gru_wx.py
python th_rnn_inference_cpu_gru_wx.py
python th_rnn_inference_cpu_gru_wx.py
python th_rnn_inference_cpu_gru_wx.py
python th_rnn_inference_cpu_gru_wx.py
