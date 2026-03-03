import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

class SMILESDataset(Dataset):
    def __init__(self, csv_file):
        """
        TODO 1: 
        - pandas를 이용해 csv_file을 읽어오세요.
        - SMILES 데이터와 Label 데이터를 클래스 속성(self.smiles_list, self.labels)으로 저장하세요.
        - 주의: 여기서 핑거프린트 변환을 미리 수행하면 안 됩니다!
        """
        df=pd.read_csv(csv_file)
        self.smiles_list=df['SMILES'].tolist()
        self.labels=df['Label'].tolist()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        TODO 2:
        - idx에 해당하는 SMILES 문자열과 Label을 가져옵니다.
        - RDKit을 사용하여 SMILES를 분자 객체(Mol)로 변환하세요.
        - 분자 객체를 반경 2(radius=2), 2048차원(nBits=2048)의 Morgan Fingerprint로 변환하세요.
        - 변환된 Fingerprint를 float32 타입의 PyTorch Tensor로, Label은 int64 타입의 Tensor로 반환하세요.
        """
        smiles=self.smiles_list[idx]
        label=self.labels[idx]
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_tensor=torch.zeros(2048, dtype=torch.float32)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            fp_tensor = torch.from_numpy(fp_array)
        label_tensor = torch.tensor(label, dtype=torch.int64)
        return fp_tensor, label_tensor

# 실행 테스트용 코드 (수정하지 마세요)
if __name__ == "__main__":
    dataset = SMILESDataset("data/raw/dummy_smiles_gem_ver.csv")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (fp, label) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Fingerprint shape: {fp.shape}, dtype: {fp.dtype}")
        print(f"Label shape: {label.shape}, dtype: {label.dtype}\n")
        break # 첫 배치만 확인