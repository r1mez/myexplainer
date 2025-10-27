from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles('N#CC(C#N)=Cc1ccc([N+](=O)[O-])o1')  # 分子表示


submol = Chem.MolFromSmarts('c1ccco1')  # 子图分子表示


matches = mol.GetSubstructMatches(submol)
print(f"找到 {len(matches)} 个匹配")


removed = Chem.ReplaceSubstructs(mol, submol, Chem.MolFromSmarts('c1ccco1'), replaceAll=True)
# removed = Chem.DeleteSubstructs(mol, submol)


result = removed[0]

print("原始分子:", Chem.MolToSmiles(mol))
print("删除羧基后:", Chem.MolToSmiles(result))
print(Chem.MolToSmiles(mol)+'.'+Chem.MolToSmiles(result))