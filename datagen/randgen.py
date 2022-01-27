import random
import pandas as pd
import copy

AA = list("ARNDBCEQZGHILKMFPSTWYV")
DDIR = "../../pdb/bylen/"
ODIR = "../../pdb/processed/"
DATA_LEN = 2000
RESIDUE_LEN = 214

data = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(RESIDUE_LEN), sep=',')["seq"]
data = data.apply(list).to_list()

# Shuffled protein sequence
shuffled_data = []
for sequence in data:
    shuffled_data.append(random.sample(sequence, RESIDUE_LEN))

# N substituted protein sequence
Ns = [10, 15, 20]
sub_data = []
for sequence in data:
    for n in Ns:
        indices = random.sample(range(RESIDUE_LEN), n)
        seq = copy.deepcopy(sequence)
        for index in indices:
            seq[index] = random.choice(AA)
        sub_data.append(seq)

# N continuous substituted protein sequence
NCs = [10, 15, 20]
con_sub_data = []
for sequence in data:
    for n in NCs:
        s = random.randint(0, RESIDUE_LEN-n-1)
        indices = range(s, s+n)
        seq = copy.deepcopy(sequence)
        for index in indices:
            seq[index] = random.choice(AA)
        con_sub_data.append(seq)

# Determine the size and print
DATA_LEN = min(len(shuffled_data), min(len(sub_data), len(con_sub_data)))
print("Size of shuffled data: {}".format(len(shuffled_data)))
print("Size of subsituted data: {}".format(len(sub_data)))
print("Size of continuously subsituted data: {}".format(len(con_sub_data)))
print("Size of random data: {}".format(DATA_LEN))

# Completely random data
rand_data = []
for i in range(DATA_LEN):
    rand_data.append(random.choices(AA, k=RESIDUE_LEN))

# Convert to Dataframe and match size
shuffled_data = pd.DataFrame(shuffled_data).sample(n=DATA_LEN)
sub_data = pd.DataFrame(sub_data).sample(n=DATA_LEN)
con_sub_data = pd.DataFrame(con_sub_data).sample(n=DATA_LEN)
rand_data = pd.DataFrame(rand_data).sample(n=DATA_LEN)

# Concat and shuffle
rand_all = pd.concat([shuffled_data, sub_data, con_sub_data, rand_data])
rand_all["label"] = 0
rand_all = rand_all.sample(frac=1)

# Save and print
rand_all.to_csv(ODIR+"rand_all.csv", index=False)
print("Random data saved. Size is {}".format(rand_all.shape[0]))