import torch
import numpy as np
from module.agents import trAgent
from module.evaluators import PQ_KL
from utils import *
from args import get_args_hallucinate
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


def main():
    # Process arguments
    args = get_args_hallucinate()

    # Amino acids to use
    aa_valid = np.arange(20)
    if args.RM_AA != "":
        aa_skip = aa2idx(args.RM_AA.replace(',',''))
        aa_valid = np.setdiff1d(aa_valid, aa_skip)

    # Initialize target sequence and length
    if args.SEQ != "":
        length = len(args.SEQ)
        target = args.SEQ
    else:
        length = args.LEN
        target = np.random.choice(aa_valid, length)

    # Get saved model path
    if '&' in args.TRDIR:
        TRDIR = args.TRDIR.split('&')
    if '&' in args.BKDIR:
        BKDIR = args.BKDIR.split('&')

    # Schedule
    tmp = args.SCHED.split(',')
    beta, NUM_EPOCHS, multiplier, schedule = 1/float(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3])
    LOG_EPOCHS = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = trAgent(target=target, aa=aa_valid, beta=beta, multiplier=multiplier, schedule=schedule)
    criterion = PQ_KL(pred_dirs=TRDIR, bkgrd_dirs=BKDIR, aa_weight=args.AA_WEIGHT, device=device)

    for epoch in range(NUM_EPOCHS):
        sequence = agent.mutate()
        input = seq_to_in(sequence)
        loss = criterion.evaluate(input)
        sequence = agent.report(loss)

        if epoch % LOG_EPOCHS == 0:
            print("EPOCHS %d/%d | SEQUENCE %s | LOSS %.6f"%(epoch, NUM_EPOCHS, sequence, loss))

if __name__ == '__main__':
    main()