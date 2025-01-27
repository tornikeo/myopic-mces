from itertools import product

from tqdm.cli import tqdm
from myopic_mces import MCES
import pandas as pd

if __name__ == '__main__':
    smiles_all = pd.read_csv('MassSpecGym.tsv', sep='\t').smiles.unique()
    N = 32
    smiles_a = smiles_all[:N]
    smiles_b = smiles_all[N:2 * N]
    total = len(smiles_a) * len(smiles_b)
    for a,b in tqdm(product(smiles_a, smiles_b)):
        MCES(a,b, solver_options=dict(msg=False))