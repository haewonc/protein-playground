import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

DDIR = "../../pdb/pdb_seqres.txt"
ODIR = "../../pdb/bylen/"
pdb = open(DDIR, 'r').readlines()

pdb = [line[:-1] for line in pdb]
names = pdb[0::2]
seqs = pdb[1::2]
lengths = [int(line.split(" ")[2][7:]) for line in names]
names = [line.split(":")[0][1:].split(" ")[0] for line in names]

# lengths_ = pd.Series(lengths)
# print(lengths_.describe())

data = [(name, seq, length) for name, seq, length in zip(names, seqs, lengths) if length>=90 and length<=335] # 20%~75% cut
names, seqs, lengths = zip(*data)

# plt.hist(lengths, bins=10)
# plt.title('Length of PDB sequences (25%-75% cut)')
# plt.xlabel('Sequence length')
# plt.ylabel('Number of sequences')
# plt.savefig('length.png')

seq_by_length = {}

for i in range(90, 336):
    seq_by_length[i] = []

for name, seq, length in zip(names, seqs, lengths):
    seq_by_length[length].append({
        "name": name,
        "seq": seq
    })

seq_counts = []
for i in range(90, 336):
    save_data = pd.DataFrame(seq_by_length[i]).drop_duplicates(subset='seq')
    seq_counts.append(save_data.shape[0])
    save_data.to_csv(ODIR+"pdb_seq_{}.csv".format(i), index=False)

# seq = pd.Series(seq_counts)
# print(seq)
# print(seq.describe()) 

max_data_length = np.argmax(seq_counts) + 90
print("Argmax number of sequences is {}".format(max_data_length))