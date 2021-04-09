

!conda install -q -y -c rdkit rdkit=2020_03_6

from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from pathlib import Path


def normalize_inchi(inchi):
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return inchi
    else:
        try:
            return Chem.MolToInchi(mol)
        except:
            return inchi


# Segfault in rdkit taken care of, run it with:
# while [ 1 ]; do python normalize_inchis.py && break; done
if __name__ == '__main__':
    # Input
    submission_name = 'submission'
    norm_path = Path(submission_name + '_norm.csv')

    # Do the job
    N = norm_path.read_text().count('\n') if norm_path.exists() else 0
    print(N, 'number of predictions already normalized')

    r = open(submission_name + '.csv', 'r')
    write_mode = 'w' if N == 0 else 'a'
    w = open(str(norm_path), write_mode, buffering=1)

    for _ in range(N):
        r.readline()
    line = r.readline()  # this line is the header or is where it died last time
    w.write(line)

    pbar = tqdm()
    while True:
        line = r.readline()
        if not line:
            break  # done
        image_id = line.split(',')[0]
        inchi = ','.join(line[:-1].split(',')[1:]).replace('"', '')
        inchi_norm = normalize_inchi(inchi)
        w.write(f'{image_id},"{inchi_norm}"\n')
        pbar.update(1)

    r.close()
    w.close()



import pandas as pd
import edlib
from tqdm import tqdm

sub_df = pd.read_csv('../input/notebook1ff4bb04fe/submission.csv')
sub_norm_df = pd.read_csv('submission_norm.csv')

lev = 0
N = len(sub_df)
for i in tqdm(range(N)):
    inchi, inchi_norm = sub_df.iloc[i,1], sub_norm_df.iloc[i,1]
    lev += edlib.align(inchi, inchi_norm)['editDistance']

print(lev/N)