# Job Description (JD) Processing Guide

本指南说明如何使用 `extract_skills_from_jd.ipynb` 批量处理大量parquet文件，避免内存溢出。

## 🎯 问题背景

**原始问题：**
- 有一个几十GB的大parquet文件
- 已拆分成200-300个小parquet文件
- 一次性读取所有文件会导致内存溢出(OOM)

**解决方案：**
- 批量处理文件（每次处理几个文件）
- 在每个文件内部分块处理（chunk processing）
- 增量保存结果
- 支持断点续传

## 📁 文件结构

推荐的目录结构：

```
SkillExtract/
├── JD/                              # 输入：小parquet文件
│   ├── jd_part_001.parquet
│   ├── jd_part_002.parquet
│   ├── ...
│   └── jd_part_300.parquet
├── output/
│   └── extracted_skills/           # 输出：处理后的文件
│       ├── jd_part_001_processed.parquet
│       ├── jd_part_002_processed.parquet
│       ├── ...
│       └── processed_files.txt     # 检查点文件
└── extract_skills_from_jd.ipynb   # 主处理notebook
```

## 🚀 快速开始

### 1. 准备数据

将你的parquet文件放在 `../JD/` 目录（notebook上一层的JD文件夹）：

```bash
# 创建JD目录
mkdir -p ../JD

# 移动或复制parquet文件到JD目录
mv your_parquet_files/*.parquet ../JD/
```

### 2. 配置参数

打开 `extract_skills_from_jd.ipynb`，修改配置：

```python
# ============================================================================
# CONFIGURATION - 根据你的需求修改这些参数
# ============================================================================

INPUT_DIR = Path("../JD")           # 输入目录
OUTPUT_DIR = Path("./output/extracted_skills")  # 输出目录

BATCH_SIZE = 10                     # 每次处理多少个文件（根据内存调整）
ROWS_PER_CHUNK = 1000               # 每个文件内部每次处理多少行

TEXT_COLUMN = 'job_description'     # 包含文本的列名
ID_COLUMN = 'job_id'                # ID列名（如果有）
```

### 3. 运行Notebook

按顺序执行notebook的各个cell：

1. **导入库** → 加载必要的Python包
2. **配置参数** → 设置输入输出路径
3. **加载SkillNER** → 初始化技能提取模型
4. **发现文件** → 扫描所有parquet文件
5. **批量处理** → 开始处理（这是主要步骤）
6. **查看统计** → 检查处理结果

## ⚙️ 参数调优

### 内存优化

如果仍然内存不足，按以下顺序调整：

#### 选项1: 减少批次大小
```python
BATCH_SIZE = 5  # 从10减到5
# 甚至可以设为1，一次只处理一个文件
```

#### 选项2: 减少chunk大小
```python
ROWS_PER_CHUNK = 500  # 从1000减到500
# 更极端的情况可以设为100
```

#### 选项3: 只读取需要的列
修改 `process_parquet_file_in_chunks` 函数：

```python
# 原来：读取所有列
pf = pd.read_parquet(filepath)

# 改为：只读取需要的列
pf = pd.read_parquet(filepath, columns=[TEXT_COLUMN, ID_COLUMN])
```

### 性能优化

如果内存充足，想加快速度：

```python
BATCH_SIZE = 20          # 增加批次大小
ROWS_PER_CHUNK = 5000    # 增加chunk大小
```

## 🔄 断点续传

### 功能说明

如果处理过程被中断（断电、错误、手动停止），可以从上次停止的地方继续：

```python
RESUME_FROM_CHECKPOINT = True  # 设为True启用续传
```

### 工作原理

- 每个文件处理完成后，文件名会写入 `processed_files.txt`
- 重新运行时，会跳过已在 `processed_files.txt` 中的文件
- 只处理剩余未完成的文件

### 重新开始

如果想从头开始处理所有文件：

```python
RESUME_FROM_CHECKPOINT = False  # 禁用续传

# 或者删除检查点文件
rm output/extracted_skills/processed_files.txt
```

## 📊 输出格式

### 单个输出文件

每个输入文件会生成一个对应的输出文件：

**输入：** `jd_part_001.parquet`
**输出：** `jd_part_001_processed.parquet`

输出文件包含原始列 + 新增列：

| 列名 | 说明 |
|------|------|
| 原始列... | 保留所有原始数据 |
| `extracted_skills` | 提取的技能列表（list） |
| `num_skills` | 提取的技能数量（int） |

### 示例

```python
# 读取一个输出文件
df = pd.read_parquet('output/extracted_skills/jd_part_001_processed.parquet')

# 查看结果
print(df[['job_id', 'extracted_skills', 'num_skills']].head())

# 输出示例：
#   job_id    extracted_skills              num_skills
# 0  JD001     [Python, SQL, Git]            3
# 1  JD002     [Java, Spring Boot]           2
# 2  JD003     []                            0
```

## 🔍 监控处理进度

### 方法1: 查看控制台输出

notebook会实时显示：

```
Batch 1/30: Processing files 1-10
----------------------------------------------------------------------
✓ Processed jd_part_001.parquet: 5000 rows, saved to jd_part_001_processed.parquet
✓ Processed jd_part_002.parquet: 4800 rows, saved to jd_part_002_processed.parquet
...
✓ Batch 1/30 completed
======================================================================
```

### 方法2: 检查输出目录

```bash
# 查看已处理的文件数量
ls output/extracted_skills/*_processed.parquet | wc -l

# 查看检查点文件
cat output/extracted_skills/processed_files.txt
```

### 方法3: 在notebook中查询

```python
processed = len(get_processed_files())
total = len(all_parquet_files)
print(f"进度: {processed}/{total} ({100*processed/total:.1f}%)")
```

## 💡 最佳实践

### 1. 先测试小批次

首次运行时，先用小批次测试：

```python
BATCH_SIZE = 2
ROWS_PER_CHUNK = 100

# 处理完2个文件后，检查结果
# 如果正常，再增加批次大小
```

### 2. 监控内存使用

运行 "Memory Usage Check" cell 监控内存：

```python
# 在notebook中运行
import psutil
process = psutil.Process(os.getpid())
print(f"当前内存: {process.memory_info().rss / (1024**3):.2f} GB")
```

### 3. 定期检查输出

每处理一批文件后，检查输出质量：

```python
# 随机抽查一个文件
df = pd.read_parquet(random.choice(list(OUTPUT_DIR.glob('*_processed.parquet'))))
print(f"平均技能数: {df['num_skills'].mean():.2f}")
print(f"有技能的比例: {(df['num_skills'] > 0).mean():.2%}")
```

### 4. 保留原始文件

处理完成后，不要立即删除原始parquet文件：

- 先验证所有输出文件都正确生成
- 抽查几个文件确保数据完整
- 确认无误后再清理原始文件

## ⚠️ 常见问题

### Q1: 仍然内存溢出怎么办？

**A:** 逐步降低参数：

```python
# 步骤1: 减少批次
BATCH_SIZE = 1  # 一次只处理1个文件

# 步骤2: 减少chunk
ROWS_PER_CHUNK = 100  # 每次只处理100行

# 步骤3: 只读需要的列
pf = pd.read_parquet(filepath, columns=['job_description', 'job_id'])
```

### Q2: 处理太慢怎么办？

**A:** 几个优化方向：

1. **增加批次大小**（如果内存允许）
2. **使用多进程**（需要修改代码，使用 `multiprocessing`）
3. **使用GPU加速**（如果SkillNER支持）
4. **使用云服务器**（更多CPU核心）

### Q3: 某个文件处理失败怎么办？

**A:** Notebook会跳过失败的文件并继续：

```
✗ Error processing jd_part_050.parquet: [错误信息]
```

可以：
1. 查看错误信息
2. 单独检查该文件
3. 修复后重新处理（将其从 `processed_files.txt` 中删除）

### Q4: 如何合并所有输出文件？

**A:** 有两种方式：

**方式1: 在notebook中合并（需要足够内存）**

```python
# 设置标志
COMBINE_RESULTS = True

# 运行"Optional: Combine Results" cell
```

**方式2: 使用命令行（推荐，更省内存）**

```python
# 使用pandas逐个读取并写入
import pandas as pd
from pathlib import Path

output_files = list(Path('output/extracted_skills').glob('*_processed.parquet'))

# 写入第一个文件
df_combined = pd.read_parquet(output_files[0])
df_combined.to_parquet('all_skills.parquet', index=False)

# 追加其余文件
for f in output_files[1:]:
    df = pd.read_parquet(f)
    df.to_parquet('all_skills.parquet', append=True)
```

### Q5: 可以在云端运行吗？

**A:** 可以！推荐平台：

- **Google Colab**: 免费，但内存有限（12GB）
- **Kaggle Notebooks**: 免费，30GB内存
- **AWS SageMaker**: 付费，可选大内存实例
- **Azure ML**: 付费，灵活配置

上传notebook和数据到云端，然后运行即可。

## 📈 性能参考

基于不同配置的大致性能（仅供参考）：

| 配置 | 内存使用 | 处理速度 | 适用场景 |
|------|---------|---------|---------|
| BATCH=1, CHUNK=100 | ~2GB | 慢 | 内存极小 |
| BATCH=5, CHUNK=500 | ~4GB | 中等 | 普通笔记本 |
| BATCH=10, CHUNK=1000 | ~8GB | 快 | 推荐配置 |
| BATCH=20, CHUNK=5000 | ~16GB+ | 很快 | 服务器 |

**实际速度取决于：**
- CPU性能
- 硬盘读写速度（SSD vs HDD）
- parquet文件压缩率
- 文本长度和复杂度

## 🎓 进阶用法

### 使用Dask处理超大数据

如果数据量非常大（TB级别），可以使用Dask：

```python
import dask.dataframe as dd

# 读取所有parquet文件
ddf = dd.read_parquet('../JD/*.parquet')

# 使用map_partitions处理
def extract_skills_from_partition(df):
    df['extracted_skills'] = df['job_description'].apply(lambda x: extract_skills(x))
    return df

ddf = ddf.map_partitions(extract_skills_from_partition)

# 保存结果
ddf.to_parquet('output/extracted_skills_dask/')
```

### 并行处理多个文件

使用 `joblib` 并行处理：

```python
from joblib import Parallel, delayed

def process_single_file(filepath):
    # ... 处理逻辑 ...
    return result

# 并行处理
results = Parallel(n_jobs=4)(  # 使用4个CPU核心
    delayed(process_single_file)(f)
    for f in files_to_process
)
```

## 📧 获取帮助

遇到问题？

1. **检查错误信息**: 仔细阅读控制台输出的错误
2. **查看日志**: 查看 `processed_files.txt` 确认进度
3. **减小规模测试**: 用1-2个文件测试
4. **检查数据格式**: 确保parquet文件格式正确

---

**祝处理顺利！🚀**
