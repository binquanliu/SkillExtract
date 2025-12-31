# 在 Jupyter Notebook 中使用 ONET 技能提取

本指南展示如何在 Jupyter Notebook 中直接设置 input/output 路径进行技能提取。

## 🚀 快速开始

### 方法 1: 使用便捷函数（最简单）

```python
from skillner.onet_converter import extract_and_save, load_and_extract_skills

# 步骤 1: 转换 ONET JSON 为知识库
kb = extract_and_save(
    input_json='your_onet_data.json',      # 你的 ONET JSON 文件
    output_pkl='.skillner-kb/ONET_EN.pkl'  # 输出路径
)

# 步骤 2: 从文本中提取技能
skills = load_and_extract_skills(
    text='I have experience with Microsoft Excel and problem solving',
    kb_path='.skillner-kb/ONET_EN.pkl'
)

print(skills)
# ['Microsoft Excel', 'problem solving']
```

### 方法 2: 使用 Jupyter Notebook 模板

打开 `notebooks/onet_skill_extraction_demo.ipynb`，直接修改里面的配置：

```python
# 修改这些变量
input_json_path = 'your_onet_data.json'
output_pkl_path = '.skillner-kb/ONET_EN.pkl'
```

---

## 📖 详细使用指南

### 1. 转换 ONET JSON 到知识库

#### 选项 A: 一行代码完成

```python
from skillner.onet_converter import extract_and_save

# 转换并保存
kb = extract_and_save(
    input_json='my_onet.json',
    output_pkl='.skillner-kb/ONET_EN.pkl'
)
```

#### 选项 B: 自定义 sections

```python
kb = extract_and_save(
    input_json='my_onet.json',
    output_pkl='.skillner-kb/ONET_EN.pkl',
    sections=['Skills', 'Technology Skills'],  # 只提取这些部分
    verbose=True  # 显示进度
)
```

#### 选项 C: 使用 notebook 中的函数

在 Jupyter cell 中直接定义和使用：

```python
import json
import pickle
from pathlib import Path

# 设置路径
INPUT_JSON = 'your_onet.json'
OUTPUT_PKL = '.skillner-kb/ONET_EN.pkl'

# 读取 JSON
with open(INPUT_JSON, 'r') as f:
    onet_data = json.load(f)

# 提取技能（使用 notebook 中的函数）
from skillner.onet_converter import extract_from_occupation, merge_knowledge_bases

if isinstance(onet_data, list):
    kbs = [extract_from_occupation(occ) for occ in onet_data]
    final_kb = merge_knowledge_bases(kbs)
else:
    final_kb = extract_from_occupation(onet_data)

# 保存
Path(OUTPUT_PKL).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(final_kb, f)

print(f"✓ Saved {len(final_kb)} skills to {OUTPUT_PKL}")
```

---

### 2. 从文本中提取技能

#### 选项 A: 最简单的方式

```python
from skillner.onet_converter import load_and_extract_skills

# 一行提取
skills = load_and_extract_skills(
    text='I have skills in Microsoft Excel and critical thinking',
    kb_path='.skillner-kb/ONET_EN.pkl'
)

print(skills)
# ['Microsoft Excel', 'critical thinking']
```

#### 选项 B: 获取详细信息

```python
skills_detailed = load_and_extract_skills(
    text='I have skills in Microsoft Excel and critical thinking',
    kb_path='.skillner-kb/ONET_EN.pkl',
    return_details=True  # 返回详细信息
)

for skill in skills_detailed:
    print(f"Skill: {skill['skill']}")
    print(f"  Section: {skill['section']}")
    print(f"  Occupation: {skill['occupation']}")
```

#### 选项 C: 批量处理多个文本

```python
from skillner.onet_converter import extract_skills_batch

resumes = [
    'Resume 1: Experience with Excel and problem solving...',
    'Resume 2: Skills in critical thinking and coordination...',
    'Resume 3: Proficient in QuickBooks and financial management...'
]

all_skills = extract_skills_batch(
    texts=resumes,
    kb_path='.skillner-kb/ONET_EN.pkl'
)

for i, skills in enumerate(all_skills, 1):
    print(f"Resume {i}: {skills}")
```

---

### 3. 合并多个知识库

```python
from skillner.onet_converter import merge_kb_files

# 合并 ESCO 和 ONET
merged_kb = merge_kb_files(
    kb_paths=[
        '.skillner-kb/ESCO_EN.pkl',
        '.skillner-kb/ONET_EN.pkl'
    ],
    output_path='.skillner-kb/MERGED_EN.pkl'
)

print(f"Merged KB has {len(merged_kb)} skills")
```

---

### 4. 查看知识库统计信息

```python
from skillner.onet_converter import inspect_kb, get_kb_stats

# 显示知识库信息
inspect_kb('.skillner-kb/ONET_EN.pkl', n_samples=10)

# 获取统计数据
stats = get_kb_stats('.skillner-kb/ONET_EN.pkl')
print(stats)
# {
#     'total_skills': 5432,
#     'total_entries': 8901,
#     'by_section': {'Skills': 2341, 'Technology Skills': 3123, ...},
#     'file_size_kb': 1234.5
# }
```

---

## 🎯 完整示例

### 示例 1: 从头到尾的完整流程

```python
from skillner.onet_converter import extract_and_save, load_and_extract_skills

# 1. 转换 ONET JSON
print("Step 1: Converting ONET JSON...")
kb = extract_and_save(
    input_json='onet_data.json',
    output_pkl='.skillner-kb/ONET_EN.pkl'
)

# 2. 分析简历
print("\nStep 2: Analyzing resume...")
resume = """
Senior Software Engineer with 8 years of experience.
Strong skills in:
- Critical thinking and problem solving
- Microsoft Excel, Word, and PowerPoint
- Coordination and communication
"""

skills = load_and_extract_skills(resume, '.skillner-kb/ONET_EN.pkl')

# 3. 显示结果
print("\nExtracted skills:")
for skill in skills:
    print(f"  ✓ {skill}")
```

### 示例 2: 使用 Pandas 分析

```python
import pandas as pd
from skillner.onet_converter import load_and_extract_skills

# 准备数据
candidates = {
    'Alice': 'Experience with Excel, critical thinking, and problem solving',
    'Bob': 'Skills in QuickBooks, financial management, and coordination',
    'Carol': 'Proficient in Microsoft Office, communication, and leadership'
}

# 提取技能
results = []
for name, resume in candidates.items():
    skills = load_and_extract_skills(resume, '.skillner-kb/ONET_EN.pkl')
    for skill in skills:
        results.append({'Name': name, 'Skill': skill})

# 创建 DataFrame
df = pd.DataFrame(results)

# 分析
print("\nSkills by candidate:")
print(df.groupby('Name')['Skill'].apply(list))

print("\nMost common skills:")
print(df['Skill'].value_counts().head(5))
```

### 示例 3: 可视化技能分布

```python
import matplotlib.pyplot as plt
from skillner.onet_converter import load_and_extract_skills
from collections import Counter

# 分析多个简历
resumes = [
    'Skills: Excel, problem solving, critical thinking',
    'Experience: Word, Excel, coordination',
    'Proficient in: critical thinking, Excel, communication'
]

all_skills = []
for resume in resumes:
    skills = load_and_extract_skills(resume, '.skillner-kb/ONET_EN.pkl')
    all_skills.extend(skills)

# 统计
skill_counts = Counter(all_skills)

# 绘图
plt.figure(figsize=(10, 6))
skills, counts = zip(*skill_counts.most_common(10))
plt.barh(skills, counts)
plt.xlabel('Frequency')
plt.title('Top 10 Most Common Skills')
plt.tight_layout()
plt.show()
```

---

## ⚙️ 配置选项

### extract_and_save 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_json` | str | - | ONET JSON 文件路径 |
| `output_pkl` | str | - | 输出 pickle 文件路径 |
| `sections` | list | 所有主要 sections | 要处理的 sections 列表 |
| `verbose` | bool | True | 是否显示进度信息 |

### load_and_extract_skills 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | str | - | 要分析的文本 |
| `kb_path` | str | - | 知识库文件路径 |
| `max_window_size` | int | 5 | 技能名称最大词数 |
| `return_details` | bool | False | 是否返回详细信息 |

---

## 🔧 常见任务

### 任务 1: 只提取技术技能

```python
kb = extract_and_save(
    input_json='onet.json',
    output_pkl='tech_skills.pkl',
    sections=['Technology Skills']  # 只要技术技能
)
```

### 任务 2: 调整匹配窗口大小

```python
# 如果技能名称很长（如 "Management of Financial Resources"）
skills = load_and_extract_skills(
    text=my_text,
    kb_path='.skillner-kb/ONET_EN.pkl',
    max_window_size=6  # 增加窗口大小
)
```

### 任务 3: 保存提取结果为 JSON

```python
import json
from skillner.onet_converter import load_and_extract_skills

skills = load_and_extract_skills(
    text=my_text,
    kb_path='.skillner-kb/ONET_EN.pkl',
    return_details=True
)

# 保存为 JSON
with open('extracted_skills.json', 'w') as f:
    json.dump(skills, f, indent=2)
```

### 任务 4: 从文件夹批量处理

```python
from pathlib import Path
from skillner.onet_converter import load_and_extract_skills

# 读取所有简历文件
resume_folder = Path('resumes/')
results = {}

for resume_file in resume_folder.glob('*.txt'):
    with open(resume_file, 'r') as f:
        text = f.read()

    skills = load_and_extract_skills(text, '.skillner-kb/ONET_EN.pkl')
    results[resume_file.name] = skills

# 显示结果
for filename, skills in results.items():
    print(f"\n{filename}:")
    print(f"  Found {len(skills)} skills: {', '.join(skills)}")
```

---

## 📝 Jupyter Notebook 模板

我们提供了一个完整的 Jupyter notebook 模板：

**位置**: `notebooks/onet_skill_extraction_demo.ipynb`

包含：
1. ✅ 从 ONET JSON 提取技能
2. ✅ 使用知识库进行技能提取
3. ✅ 合并多个知识库
4. ✅ 实际应用示例
5. ✅ 数据可视化

**使用方法**:
1. 打开 notebook
2. 修改配置 cell 中的路径
3. 运行所有 cells

---

## 💡 提示和技巧

### 提示 1: 在 notebook 顶部设置路径

```python
# 在第一个 cell 中设置所有路径
ONET_JSON = 'data/onet_occupations.json'
KB_OUTPUT = '.skillner-kb/ONET_EN.pkl'
TEXT_TO_ANALYZE = """
Your text here...
"""

# 然后在后续 cells 中使用这些变量
```

### 提示 2: 缓存知识库

```python
# 避免重复加载
_kb_cache = {}

def get_kb(kb_path):
    if kb_path not in _kb_cache:
        from skillner.onet_converter import load_kb
        _kb_cache[kb_path] = load_kb(kb_path)
    return _kb_cache[kb_path]

# 使用
kb = get_kb('.skillner-kb/ONET_EN.pkl')
```

### 提示 3: 进度条（大数据集）

```python
from tqdm.notebook import tqdm
from skillner.onet_converter import load_and_extract_skills

results = []
for resume in tqdm(all_resumes, desc="Processing"):
    skills = load_and_extract_skills(resume, kb_path)
    results.append(skills)
```

---

## ❓ 常见问题

**Q: 如何在 Google Colab 中使用？**

```python
# 1. 上传文件到 Colab
from google.colab import files
uploaded = files.upload()  # 选择你的 ONET JSON

# 2. 使用
from skillner.onet_converter import extract_and_save
kb = extract_and_save('uploaded_file.json', 'output.pkl')
```

**Q: 如何处理大文件？**

分批处理：
```python
import json

# 分批读取
batch_size = 100
with open('large_onet.json', 'r') as f:
    data = json.load(f)

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # 处理 batch
```

**Q: 可以直接从字典提取吗（不保存文件）？**

可以：
```python
from skillner.onet_converter import extract_from_occupation

onet_dict = {...}  # 你的 ONET 数据
kb = extract_from_occupation(onet_dict)
# kb 就是知识库，可以直接使用
```

---

## 📚 相关资源

- [完整 Notebook 示例](../notebooks/onet_skill_extraction_demo.ipynb)
- [ONET 集成指南](ONET_INTEGRATION.md)
- [句子技能提取](SENTENCE_SKILL_EXTRACTION.md)
- [快速开始](../ONET_QUICKSTART.md)
