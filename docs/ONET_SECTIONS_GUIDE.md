# ONET Sections 完整指南

本文档说明 O*NET 数据中所有可用的 sections 及其用途。

## 📋 主要 Sections

### 1. Skills（技能）⭐ 推荐

**内容**: 核心职业技能

**示例**:
- Judgment and Decision Making
- Complex Problem Solving
- Critical Thinking
- Coordination
- Management of Financial Resources

**格式**: `技能名— 描述...`

**建议**: ✅ **必须包含**，这是最核心的技能数据

---

### 2. Abilities（能力）⭐ 推荐

**内容**: 认知、体能、感知等基础能力

**示例**:
- Deductive Reasoning
- Oral Comprehension
- Problem Sensitivity
- Written Expression
- Near Vision

**格式**: `能力名— 描述...`

**建议**: ✅ **强烈推荐**，补充 Skills，覆盖更基础的能力维度

---

### 3. Knowledge（知识）⭐ 推荐

**内容**: 领域专业知识

**示例**:
- Economics and Accounting
- Administration and Management
- Customer and Personal Service
- Mathematics
- English Language

**格式**: `知识领域— 描述...`

**建议**: ✅ **强烈推荐**，识别专业领域知识

---

### 4. Technology Skills（技术技能）⭐ 推荐

**内容**: 软件、工具、技术平台

**示例**:
- Accounting software— QuickBooks; Sage 50
- Spreadsheet software— Microsoft Excel
- Word processing software— Microsoft Word

**格式**: `类别— 工具1; 工具2; 工具3`

**建议**: ✅ **必须包含**，对技术岗位尤其重要

---

### 5. Work Activities（工作活动）

**内容**: 通用工作活动

**示例**:
- Making Decisions and Solving Problems
- Establishing and Maintaining Interpersonal Relationships
- Communicating with Supervisors, Peers, or Subordinates

**格式**: `活动名— 描述...`

**建议**: ✅ 推荐包含，识别工作行为

---

### 6. Detailed Work Activities（详细工作活动）

**内容**: 具体的工作任务

**示例**:
- Direct financial operations
- Confer with organizational members
- Prepare operational budgets

**格式**: 短句描述

**建议**: ⚠️ 可选，比较具体，可能提取到很多技能

---

### 7. Tasks（任务）

**内容**: 职业的具体任务描述

**示例**:
- Direct or coordinate an organization's financial or budget activities
- Confer with board members, organization officials, or staff members

**格式**: 完整句子

**建议**: ⚠️ 可选，句子较长，需要 NLP 提取

---

### 8. Work Context（工作环境）❌ 不推荐提取

**内容**: 工作环境特征（频率、重要性等）

**示例**:
- E-Mail— 97% responded "Every day."
- Face-to-Face Discussions— 90% responded "Every day."

**建议**: ❌ 不适合作为技能提取

---

### 9. Work Styles（工作风格）⚠️ 可选

**内容**: 个人特质和工作态度

**示例**:
- Integrity
- Dependability
- Attention to Detail
- Leadership
- Cooperation

**格式**: `特质名— 描述...`

**建议**: ⚠️ 可选，更偏向软技能/个人特质

---

### 10. Interests（兴趣）❌ 不推荐提取

**内容**: Holland 职业兴趣代码

**示例**:
- Enterprising
- Conventional
- Social

**建议**: ❌ 不适合作为技能

---

### 11. Work Values（工作价值观）❌ 不推荐提取

**内容**: 工作价值观

**示例**:
- Achievement
- Recognition
- Support

**建议**: ❌ 不适合作为技能

---

## 🎯 推荐配置

### 配置 1: 核心技能（默认，已更新）⭐ 推荐

包含最重要的技能维度：

```python
sections = [
    'Skills',              # 核心技能
    'Abilities',           # 基础能力  ← 新增
    'Knowledge',           # 领域知识  ← 新增
    'Technology Skills',   # 技术技能
    'Work Activities',     # 工作活动
    'Detailed Work Activities'  # 详细活动
]
```

**适用**: 大多数应用场景

**优点**:
- ✅ 覆盖全面（技能+能力+知识）
- ✅ 提取质量高
- ✅ 适中的数据量

---

### 配置 2: 完整提取

包含所有有用的 sections：

```python
sections = [
    'Skills',
    'Abilities',
    'Knowledge',
    'Technology Skills',
    'Work Activities',
    'Detailed Work Activities',
    'Tasks',
    'Work Styles'
]
```

**适用**: 需要最全面的技能覆盖

**优点**: ✅ 最全面
**缺点**: ⚠️ 知识库较大，可能有噪音

---

### 配置 3: 技术技能为主

只关注技术技能：

```python
sections = [
    'Technology Skills',   # 技术工具
    'Skills',             # 核心技能
    'Knowledge'           # 专业知识
]
```

**适用**: 技术岗位招聘/筛选

---

### 配置 4: 软技能为主

关注软技能和能力：

```python
sections = [
    'Skills',             # 核心技能
    'Abilities',          # 基础能力
    'Work Activities',    # 工作活动
    'Work Styles'         # 工作风格
]
```

**适用**: 管理/领导岗位

---

## 📊 各 Section 数据量对比

基于典型 ONET 数据集的统计：

| Section | 平均条目数/职业 | 提取后技能数 | 推荐度 |
|---------|----------------|------------|-------|
| Skills | 35-40 | 35-40 | ⭐⭐⭐⭐⭐ |
| Abilities | 20-30 | 20-30 | ⭐⭐⭐⭐⭐ |
| Knowledge | 10-20 | 10-20 | ⭐⭐⭐⭐⭐ |
| Technology Skills | 10-50 | 30-150 | ⭐⭐⭐⭐⭐ |
| Work Activities | 15-25 | 15-25 | ⭐⭐⭐⭐ |
| Detailed Work Activities | 20-40 | 20-40 | ⭐⭐⭐ |
| Tasks | 10-30 | 20-60 | ⭐⭐ |
| Work Styles | 8-15 | 8-15 | ⭐⭐ |

---

## 🔧 如何使用

### 使用 Python 模块

```python
from skillner.onet_converter import extract_and_save

# 默认（包含 Skills, Abilities, Knowledge, Technology Skills 等）
kb = extract_and_save('onet.json', 'output.pkl')

# 自定义 sections
kb = extract_and_save(
    input_json='onet.json',
    output_pkl='output.pkl',
    sections=['Skills', 'Abilities', 'Knowledge', 'Technology Skills']
)
```

### 使用命令行脚本

```bash
# 默认（已包含 Abilities 和 Knowledge）
python scripts/extract_skills_from_sentences.py onet.json output.pkl

# 自定义 sections
python scripts/extract_skills_from_sentences.py onet.json output.pkl \
    --sections "Skills,Abilities,Knowledge,Technology Skills"
```

---

## 💡 最佳实践

### 1. 先探索你的数据

```python
import json

with open('your_onet.json', 'r') as f:
    data = json.load(f)

# 查看第一个职业的 sections
sample = data[0] if isinstance(data, list) else data
print("Available sections:")
for section_name, section_data in sample.get('sections', {}).items():
    if section_data.get('type') == 'list':
        items = section_data.get('items', [])
        print(f"  {section_name}: {len(items)} items")
        if items:
            print(f"    Example: {items[0][:80]}...")
```

### 2. 逐步添加 sections

从核心开始，逐步添加：

```python
# 第1步：只提取核心技能
kb1 = extract_and_save('onet.json', 'kb1.pkl',
                       sections=['Skills'])

# 第2步：添加能力和知识
kb2 = extract_and_save('onet.json', 'kb2.pkl',
                       sections=['Skills', 'Abilities', 'Knowledge'])

# 第3步：添加技术技能
kb3 = extract_and_save('onet.json', 'kb3.pkl',
                       sections=['Skills', 'Abilities', 'Knowledge',
                                'Technology Skills'])

# 比较提取结果
print(f"KB1: {len(kb1)} skills")
print(f"KB2: {len(kb2)} skills")
print(f"KB3: {len(kb3)} skills")
```

### 3. 根据应用场景选择

```python
# 技术岗位
tech_sections = ['Skills', 'Abilities', 'Knowledge', 'Technology Skills']

# 管理岗位
mgmt_sections = ['Skills', 'Abilities', 'Knowledge', 'Work Activities', 'Work Styles']

# 通用
general_sections = ['Skills', 'Abilities', 'Knowledge',
                   'Technology Skills', 'Work Activities']
```

---

## 📝 更新日志

- **2025-01-01**: 默认 sections 更新，新增 **Abilities** 和 **Knowledge**
- 之前版本只包含: Skills, Technology Skills, Work Activities, Detailed Work Activities

---

## ❓ 常见问题

**Q: 为什么之前没有包含 Abilities 和 Knowledge？**

A: 这是遗漏，现在已经修复。这两个 section 非常重要！

**Q: 应该使用哪些 sections？**

A: 推荐使用默认配置（Skills, Abilities, Knowledge, Technology Skills, Work Activities, Detailed Work Activities），覆盖最全面。

**Q: Tasks section 要不要包含？**

A: 可选。Tasks 是完整句子，需要 NLP 提取，可能提取到更多技能但也可能有噪音。

**Q: 如何知道我的数据有哪些 sections？**

A: 运行上面的"探索数据"代码查看。

---

## 🎯 总结

**必须包含**:
- ✅ Skills
- ✅ Abilities（新增）
- ✅ Knowledge（新增）
- ✅ Technology Skills

**推荐包含**:
- ✅ Work Activities
- ✅ Detailed Work Activities

**可选**:
- ⚠️ Tasks
- ⚠️ Work Styles

**不推荐**:
- ❌ Work Context
- ❌ Interests
- ❌ Work Values
