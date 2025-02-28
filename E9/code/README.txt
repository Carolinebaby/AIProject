LSTM.py 是单向长度期记忆网络的实现
BiLSTM.py 是 双向长短期记忆网络的实现
LSTM_QNLI_all.py 是在完整的QNLI数据集上测试的代码

文件存放位置的架构：
dir/
│
├─ LSTM.py
├─ BiLSTM.py
├─ LSTM_QNLI_all.py
│
├─ QNLI/
│	├─ dev_40.tsv
│	└─ train_40.tsv
│
├─ QNLI_all/
│	├─ dev.tsv
│	└─ train.tsv
│
└─ glove.6B/
 	└─ glove.6B.100d.txt 