import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

class SMILESDataset(Dataset):
    def __init__(self, csv_file):
        """
        TODO 1: Pandas를 이용해 csv_file을 읽어와서 self.data에 저장하세요.
        (힌트: pd.read_csv 활용)
        """
        self.data = pd.read_csv(csv_file)# 이 부분을 채우세요.

    def __len__(self):
        # 데이터의 총 개수 반환 (이 부분은 완성되어 있음)
        return len(self.data)

    def __getitem__(self, idx):
        """
        TODO 2: 특정 인덱스(idx)의 SMILES 문자열을 가져와서 RDKit으로 2048차원 Fingerprint로 변환하고,
        PyTorch Tensor 형태로 변환하여 Label(정답)과 함께 반환하세요.
        """
        # 1. 특정 row(idx)의 SMILES와 Label 가져오기
        smiles = self.data.iloc[idx]['SMILES'] #iloc는 행과 열의 위치로 데이터를 가져오는 함수입니다. idx는 행의 위치를 나타냅니다. 'SMILES'는 열의 이름입니다.
        label = self.data.iloc[idx]['Label']
        
        # 2. RDKit을 이용해 SMILES 문자열 -> 분자 객체(Mol) 변환
        mol = Chem.MolFromSmiles(smiles)
        
        # 3. 예외 처리: 불량 SMILES 방어 코드 (매우 중요)
        if mol is None:
            # 불량 데이터면 0으로 가득 찬 텐서를 반환 (Shape: 2048)
            fp_tensor = torch.zeros(2048, dtype=torch.float32) #zeros는 주어진 shape의 배열을 0으로 채워서 생성하는 함수입니다. 여기서는 2048차원의 벡터를 생성합니다.
        else:
            # TODO 2-1: AllChem.GetMorganFingerprintAsBitVect를 이용해 분자(mol)를 Morgan FP로 변환하세요. (반경=2, nBits=2048)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) 
            # AllChem.GetMorganFingerprintAsBitVect는 RDKit에서 분자 객체를 Morgan Fingerprint로 변환하는 함수입니다. radius는 Morgan Fingerprint의 반경을 나타내며, nBits는 생성할 Fingerprint의 길이를 나타냅니다.
            # Morgan Fingerprint는 분자의 구조적 특징을 고정된 길이의 벡터로 표현하는 방법입니다. 반경이 2인 경우, 각 원자 주변의 2단계 이웃까지의 구조적 정보를 포함하게 됩니다. nBits가 2048인 경우, 생성되는 Fingerprint 벡터는 2048차원이 됩니다.
            # TODO 2-2: RDKit BitVect를 Numpy Array로 바꾼 뒤, 다시 PyTorch Tensor(dtype=float32)로 변환하세요.
            fp_array = np.zeros((2048,), dtype=np.int8)
            #np.zeros(2048,)는 행의 수가 2048이고, 열의 수가 0인 배열을 생성합니다. dtype=np.int8는 배열의 데이터 타입을 8비트 정수로 지정합니다. 이 배열은 RDKit의 BitVect를 Numpy Array로 변환하는 데 사용됩니다. 
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            fp_tensor = torch.tensor(fp_array, dtype=torch.float32) 
            #torch.tensor는 Numpy Array를 PyTorch Tensor로 변환하는 함수입니다. dtype=torch.float32는 텐서의 데이터 타입을 32비트 부동 소수점으로 지정합니다. 이렇게 변환된 fp_tensor는 모델의 입력으로 사용될 수 있습니다.
            
        # 4. Label 데이터도 PyTorch Tensor로 변환
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return fp_tensor, label_tensor

# ==========================================
# 실행 테스트 코드 (이 부분은 건드리지 마세요)
# ==========================================
if __name__ == "__main__":
    # 1. Dataset 인스턴스 생성
    dataset = SMILESDataset('data/raw/dummy_smiles.csv')
    
    # 2. DataLoader 컨베이어 벨트 생성 (한 번에 2개씩 가져오기)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. 배치 데이터 출력 테스트
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"--- Batch {batch_idx+1} ---")
        print(f"Features shape (입력 데이터): {features.shape}")
        print(f"Labels shape (정답 데이터): {labels.shape}")
        break # 첫 번째 배치만 확인하고 루프 종료