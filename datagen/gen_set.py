import random
import os 
import pandas as pd
import copy
from sklearn.model_selection import train_test_split

DDIR = "../../pdb/processed/"
ODIR = "../../pdb/exp/"
PORTIONS = [269, 359, 628]

# Prepare random sets
rand_all = pd.read_csv(DDIR+"rand_all.csv", sep=',')
rand1_train, rand1_test = train_test_split(rand_all.sample(n=897), test_size=0.3)
rand2_train, rand2_test = train_test_split(rand_all.sample(n=1281), test_size=0.3)
rand3_train, rand3_test = train_test_split(rand_all.sample(n=2178), test_size=0.3)
rand4_train, rand4_test = train_test_split(rand_all.sample(n=2691), test_size=0.3)
rand5_train, rand5_test = train_test_split(rand_all.sample(n=3588), test_size=0.3)

# Split protein
org = pd.read_csv(DDIR+"org.csv", sep=',')
org_train, org_test = train_test_split(org, test_size=0.3)

# Sample augmented data
tails = []
tags = []
links = []
for out, name in zip([tails, tags, links], ["aug_tail.csv", "aug_tag.csv", "aug_link.csv"]):
    aug_data = pd.read_csv(DDIR+name, sep=',')
    for port in PORTIONS:
        out.append(aug_data.sample(port))

# Make path
EXP_NAMES = ["trial{}".format(i) for i in range(1, 9)]
for exp_name in EXP_NAMES:
    if not os.path.isdir(ODIR+exp_name):
        os.mkdir(ODIR+exp_name)

# Generate train set for 8 experiments
train1 = pd.concat([org_train, rand1_train]).sample(frac=1)
train2 = pd.concat([org_train, tails[0], rand2_train]).sample(frac=1)
train3 = pd.concat([org_train, links[0], rand2_train]).sample(frac=1)
train4 = pd.concat([org_train, tags[0], rand2_train]).sample(frac=1)
train5 = pd.concat([tails[0], links[1], tags[0], rand2_train]).sample(frac=1)
train6 = pd.concat([tails[2], links[2], tags[2], rand4_train]).sample(frac=1)
train7 = pd.concat([org_train, tails[0], links[1], tags[0], rand3_train]).sample(frac=1)
train8 = pd.concat([org_train, tails[2], links[2], tags[2], rand5_train]).sample(frac=1)
trains = [train1, train2, train3, train4, train5, train6, train7, train8]

# Generate non-protein test set 
non_tests = [rand1_test, rand2_test, rand2_test, rand2_test, rand2_test, rand4_test, rand3_test, rand5_test]

# Save train and test sets
for exp_name, train_data, non_test_data in zip(EXP_NAMES, trains, non_tests):
    print("{} | Train size: {} Non-protein test size: {}".format(exp_name, train_data.shape[0], non_test_data.shape[0]))
    train_data.to_csv(ODIR+exp_name+"/train.csv", index=False)
    org_test.to_csv(ODIR+exp_name+"/test_ptn.csv", index=False)
    non_test_data.to_csv(ODIR+exp_name+"/test_non.csv", index=False)