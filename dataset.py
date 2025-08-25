import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# 1) 数据集与新仓库信息
dataset_id = "HuggingFaceH4/MATH-500"
new_repo_name = "top-3-MATH-500-questions"      # 你要创建的新数据集仓库名
your_username = "Youthquake123"                 # 替换成你的 HF 用户名

print(f"正在加载数据集: {dataset_id} ...")
try:
    ds = load_dataset(dataset_id)
    print("数据集加载成功！")
except Exception as e:
    print(f"加载数据集失败: {e}")
    raise SystemExit(1)

# 2) 选择分割并筛选前三题（优先 test，没有就用 train）
split = "test" if "test" in ds else ("train" if "train" in ds else None)
if split is None:
    print("该数据集没有 'test' 或 'train' 分割，请检查数据集结构。")
    raise SystemExit(1)

subset = ds[split]
if len(subset) < 3:
    print(f"{dataset_id} 的 {split} 分割少于 3 条样本（仅 {len(subset)} 条）。")
    raise SystemExit(1)

top3 = subset.select(range(3))
print(f"已从分割 '{split}' 中选出前三题。")
print(f"新数据集包含 {len(top3)} 条记录。")
print("\n新数据集第一条记录预览：")
print(top3[0])

# 3) 直接用选出的 Dataset 作为新数据集（避免 to_dict 丢失特征信息）
new_dataset: Dataset = top3

# 4) 上传到你的 Hugging Face 账户
repo_id = f"{your_username}/{new_repo_name}"
print(f"\n准备上传到: {repo_id}")

try:
    # 若想私有：private=True
    new_dataset.push_to_hub(repo_id=repo_id, private=False)
    print("数据集上传成功！")
    print(f"查看链接：https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"上传数据集失败: {e}")
    print("请确保已登录：在终端运行 `huggingface-cli login`，并确认用户名与仓库名正确。")
