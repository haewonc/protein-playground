import os
import pandas as pd 

LENGTH = 214
MAX_TAIL = 3 # Max alanine tail length
TAG_NAMES = ["T7", "V5", "S", "HAT"]
TAG_LENS = [11, 14, 15, 19]
TAG_SEQS = ["MASMTGGQQMG", "GKPIPNPLLGLDST", "KETAAAKFERQHMDS", "KDHLIHNVHKEFHAHAHNK"]
LINK_LENS = [20]
LINK_SEQS = ["GGGSGGGSGGGSGGPGS"]
LEN_RANGE = (90, 110)

DDIR = "../../pdb/bylen/"
ODIR = "../../pdb/processed/"

if not os.path.isdir(ODIR):
    os.mkdir(ODIR)

# Load sequences with fixed length
data = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(LENGTH), sep=',')["seq"]
data = pd.DataFrame(data.apply(list).to_list())
data["label"] = 1
data.to_csv(ODIR+'org.csv', index=False)
org_data_size = data.shape[0]

all_adata = None
# Load sequences to add alanine tail
for alen in range(1, MAX_TAIL+1):
    adata = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(LENGTH-alen), delimiter=',')
    adata = adata["seq"] + "A"*alen
    adata = pd.DataFrame(adata.apply(list).to_list())
    adata["label"] = 1
    if data is None:
        all_adata = adata
    else:
        all_adata = pd.concat([all_adata, adata])
all_adata.to_csv(ODIR+'aug_tail.csv', index=False)
tail_data_size = all_adata.shape[0]

all_tdata = None
# Load sequences with tag tail
for tlen, tseq in zip(TAG_LENS, TAG_SEQS):
    tdata = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(LENGTH-tlen), delimiter=',')
    tdata = tdata["seq"] + tseq
    tdata = pd.DataFrame(tdata.apply(list).to_list())
    tdata["label"] = 1
    if all_tdata is None:
        all_tdata = tdata
    else:
        all_tdata = pd.concat([all_tdata, tdata])
all_tdata.to_csv(ODIR+'aug_tag.csv', index=False)
tag_data_size = all_tdata.shape[0]

# Load sequences linked with linker
augseqs = []
for llen, lseq in zip(LINK_LENS, LINK_SEQS):
    for slen in range(90, 95):
        elen = LENGTH-llen-slen
        sdata = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(slen), delimiter=',')["seq"].tolist()
        edata = pd.read_csv(DDIR+"pdb_seq_{}.csv".format(elen), delimiter=',')["seq"].tolist()
        for s in sdata:
            for e in edata:
                augseqs.append(list(s+lseq+e))
                break
augseqs = pd.DataFrame(augseqs)
augseqs["label"] = 1
augseqs.to_csv(ODIR+'aug_link.csv', index=False)
link_data_size = augseqs.shape[0]

# Print the size of augmented dataset
print("Original dataset size: {}".format(org_data_size))
print("Alanine tail augmented dataset size: {}".format(tail_data_size))
print("Tag augmented dataset size: {}".format(tag_data_size))
print("Linker augmented dataset size: {}".format(link_data_size))