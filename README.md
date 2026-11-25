# 基于蒙特卡洛模拟的软件项目工期风险评估（MovieRecommendationSystem）

本项目为《软件工程管理》（2026）课程作业（题目 1）的配套可复现 Python 仿真源代码。以中小型全栈电影推荐系统 `MovieRecommendationSystem` 为研究案例，基于 12 个 WBS 工作包与 PERT 三点工期估算，采用 Beta-PERT（$\lambda=4$）概率分布执行 10,000 次蒙特卡洛模拟，实现项目工期概率分布估计、传统 CPM 单点对比、双维度敏感度分析（Spearman 秩相关系数与关键性指数）以及全部报告图表的自动化复现。

---

## 1. 目录结构

```text
Code/
├── README.md                  # 项目说明与运行复现指南（本文件）
├── requirements.txt           # Python 第三方依赖清单
├── main.py                    # 主仿真程序（CPM + 蒙特卡洛模拟 + 敏感度分析 + 绘图）
├── config/
│   └── wbs_tasks.csv          # 12 个工作包的依赖关系与 PERT 三点工期 (a, m, b)
└── results/                   # 运行后自动生成的仿真数据与图表目录
    ├── summary.json           # 核心统计指标汇总（均值、标准差、分位数、按期概率）
    ├── cpm_schedule.csv       # 传统 CPM 前推/后推明细表
    ├── task_sensitivity.csv   # 各工作包 Spearman 秩相关系数与关键性指数表
    ├── simulation_durations.csv # 10,000 次蒙特卡洛模拟总工期明细
    ├── task_network.png       # 图 1：WBS 任务依赖网络图
    ├── duration_histogram.png # 图 2：项目总工期概率密度分布直方图
    ├── sensitivity_spearman.png # 图 3：Spearman 秩相关敏感度排名图
    └── criticality_index.png  # 图 4：各工作包关键性指数分布图
```

---

## 2. 运行环境与依赖安装

- **Python 版本**：推荐 Python 3.9 及以上
- **依赖库**：`numpy>=1.24`、`matplotlib>=3.7`、`reportlab>=4.0`

在 `Code/` 目录下安装依赖：

```bash
pip install -r requirements.txt
```

---

## 3. 一键复现命令

执行以下命令即可 100% 复现研究报告中的全部数值结果与 4 幅图表：

```bash
python main.py --runs 10000 --seed 20261120 --deadline 53
```

### 命令行参数说明

| 参数 | 默认值 | 说明 |
| :--- | :---: | :--- |
| `--runs` | `10000` | 蒙特卡洛模拟次数（按作业要求不得少于 10,000 次） |
| `--seed` | `20261120` | 固定随机种子（内置 `xorshift32 + Marsaglia-Tsang` 跨平台可复现采样器） |
| `--deadline` | `53` | 情景截止日期（单位：工作日），用于计算按期完工概率 |

---

## 4. 预期复现基准结果

运行上述默认命令后，终端及 `results/summary.json` 将输出以下完全一致的复现结果：

- **传统 CPM 单点基准工期**：`50.17` 个工作日（关键路径：`T01 -> T02 -> T05 -> T07 -> T08 -> T09 -> T11 -> T12`）
- **蒙特卡洛模拟期望工期**：`50.40` 个工作日（标准差 `3.27` 天）
- **关键工期分位数**：`P50 = 50.35` 天、`P80 = 53.11` 天、`P90 = 54.66` 天、`P95 = 55.87` 天
- **53 个工作日情景按期完工概率**：`79.03%`
- **敏感度排名前三的高风险工作包**：
  1. `T11 前端集成与系统测试`（`rho = 0.483`, `CI = 100.0%`）
  2. `T09 在线推荐功能开发与接入`（`rho = 0.477`, `CI = 100.0%`）
  3. `T08 用户互动与社区模块开发`（`rho = 0.368`, `CI = 100.0%`）
