# SkillNER Import错误修复指南

## ❌ 错误信息

```python
ImportError: cannot import name 'Pipeline' from 'skillner'
```

## 🔍 问题原因

**错误的导入方式:**
```python
from skillner import Pipeline  # ❌ 错误！
```

**正确的导入方式:**
```python
from skillner.core import Pipeline  # ✅ 正确！
```

SkillNER的`Pipeline`类在`skillner.core`模块中，不是直接在`skillner`包中。

## ✅ 完整修复方案

### 方案1: 使用修复后的Notebook（推荐）

我已经创建了完全修复的版本：

```bash
# 使用新的修复版notebook
jupyter notebook extract_skills_from_jd_fixed.ipynb
```

**新notebook的特点:**
- ✅ 正确的SkillNER API使用
- ✅ 自动查找知识库文件
- ✅ 包含测试cell验证功能
- ✅ 详细的错误提示
- ✅ 支持顺序和并行两种模式

### 方案2: 手动修改现有代码

如果你想修改现有代码，按以下步骤：

#### 步骤1: 修改导入语句

**原来的代码:**
```python
from skillner import Pipeline
```

**改为:**
```python
from skillner.core import Pipeline, Document
from skillner.text_loaders import StrTextLoader
from skillner.matchers import SlidingWindowMatcher
from skillner.conflict_resolvers import SpanProcessor
```

#### 步骤2: 下载知识库（首次必须）

```bash
# 在终端运行
skillner-download ESCO_EN
```

这会下载技能知识库到默认位置（通常是 `~/.skillner/`）

#### 步骤3: 加载知识库

```python
import json
from pathlib import Path

# 加载知识库
kb_file = Path.home() / ".skillner" / "skill_db.json"
with open(kb_file, 'r') as f:
    skill_db = json.load(f)

# 创建查询函数
def query_skills(text: str):
    text_lower = text.lower()
    return skill_db.get(text_lower, [])
```

#### 步骤4: 修改技能提取函数

**原来的代码（不正确）:**
```python
def process_text(text, nlp):
    annotations = nlp(text)
    skills = [ent['doc_node_value'] for ent in annotations['results']['full_matches']]
    return skills
```

**改为（正确）:**
```python
def extract_skills_from_text(text):
    if pd.isna(text) or text == '':
        return []

    try:
        # 创建文档
        doc = Document()

        # 创建pipeline
        pipeline = Pipeline()

        # 添加文本加载器
        pipeline.add_node(
            StrTextLoader(str(text)),
            name='loader'
        )

        # 添加技能匹配器
        pipeline.add_node(
            SlidingWindowMatcher(
                query_skills,  # 使用我们的查询函数
                max_window_size=4,
                pre_filter=lambda word: word.lower()
            ),
            name='matcher'
        )

        # 添加冲突解决器
        pipeline.add_node(
            SpanProcessor(
                dict_filters={
                    \"max_candidate\": lambda span: max(span.li_candidates, key=len)
                }
            ),
            name=\"conflict_resolver\"
        )

        # 运行pipeline
        pipeline.run(doc)

        # 提取技能文本
        skills = []
        for sentence in doc:
            for span in sentence.li_spans:
                if 'max_candidate' in span.metadata:
                    max_candidate = span.metadata['max_candidate']
                    skill_text = \" \".join(sentence[max_candidate.window])
                    skills.append(skill_text)

        return skills

    except Exception as e:
        return []
```

## 📋 完整检查清单

运行之前检查：

- [ ] ✅ 已安装SkillNER: `pip install skillner`
- [ ] ✅ 已下载知识库: `skillner-download ESCO_EN`
- [ ] ✅ 使用正确的导入: `from skillner.core import Pipeline`
- [ ] ✅ 已加载知识库JSON文件
- [ ] ✅ 创建了正确的pipeline
- [ ] ✅ TEXT_COLUMN设置正确

## 🧪 测试代码

运行此代码测试是否正确：

```python
import sys
from pathlib import Path

# 添加路径
parent_dir = Path.cwd().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 测试导入
try:
    from skillner.core import Pipeline, Document
    from skillner.text_loaders import StrTextLoader
    from skillner.matchers import SlidingWindowMatcher
    from skillner.conflict_resolvers import SpanProcessor
    print("✓ 导入成功!")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 测试知识库
import json
kb_file = Path.home() / ".skillner" / "skill_db.json"
if kb_file.exists():
    with open(kb_file, 'r') as f:
        skill_db = json.load(f)
    print(f"✓ 知识库加载成功! 包含 {len(skill_db)} 个技能")
else:
    print(f"❌ 知识库不存在: {kb_file}")
    print("   请运行: skillner-download ESCO_EN")
```

如果所有测试都显示 ✓，那么可以继续了！

## 🚀 快速开始（使用修复版）

```bash
# 1. 确保知识库已下载
skillner-download ESCO_EN

# 2. 打开修复版notebook
jupyter notebook extract_skills_from_jd_fixed.ipynb

# 3. 按顺序运行所有cells
```

## 📊 配置建议（A100服务器）

使用修复版notebook时，可以调整这些参数：

```python
# 基本配置
BATCH_SIZE = 50          # 一次处理50个文件
ROWS_PER_CHUNK = 10000   # 每chunk处理1万行

# A100服务器推荐
USE_PARALLEL = True      # 启用并行
N_WORKERS = 16           # 16个worker

# 如果内存充足，可以更激进
BATCH_SIZE = 100
N_WORKERS = 32
```

## ⚠️ 常见问题

### Q1: 仍然报ImportError

**检查:**
```bash
# 确认skillner已安装
pip show skillner

# 重新安装
pip uninstall skillner
pip install skillner
```

### Q2: 知识库找不到

**手动指定路径:**
```python
# 查找知识库
import os
os.system("find ~ -name 'skill_db.json' 2>/dev/null")

# 如果找到了，手动指定
kb_file = Path("/path/to/skill_db.json")
```

### Q3: 提取不到技能

**可能原因:**
1. 知识库语言不匹配（ESCO_EN是英文）
2. TEXT_COLUMN列名不对
3. JD文本格式问题

**测试:**
```python
# 用英文测试
test_text = "Looking for Python developer with SQL experience"
skills = extract_skills_from_text(test_text)
print(skills)  # 应该能提取到 Python, SQL
```

### Q4: 速度太慢

**单个文本提取很慢是正常的**，因为SkillNER需要：
1. 创建Document对象
2. 构建Pipeline
3. 运行多个步骤

**优化建议:**
- 使用并行处理（`USE_PARALLEL = True`）
- 增加CHUNK大小（如果内存充足）
- 考虑缓存pipeline（高级用法）

## 📈 性能对比

| 模式 | 速度 | 内存 | 推荐场景 |
|------|------|------|---------|
| 顺序处理 | 慢 | 低 | 首次测试 |
| 并行16 workers | 快10-15x | 中 | A100标准 |
| 并行32 workers | 快20-30x | 高 | A100激进 |

## 🎯 最终建议

1. **首次运行**: 使用`extract_skills_from_jd_fixed.ipynb`的顺序模式
2. **测试成功后**: 切换到并行模式
3. **大规模处理**: 根据服务器配置调整BATCH_SIZE和N_WORKERS

---

**修复版notebook路径:**
```
extract_skills_from_jd_fixed.ipynb  ← 使用这个！
```

**原版notebook（有问题）:**
```
extract_skills_from_jd.ipynb  ← 不要用这个
```
