
import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, PageBreak
)


# ==================================================
# 1. 项目 WBS
# ==================================================

TASKS_DEFAULT = [
    ("T01", "需求分析与范围确认", "Requirements", "", 3, 4, 6),
    ("T02", "系统架构与技术方案设计", "Architecture", "T01", 3, 4, 7),
    ("T03", "数据库与数据模型设计", "Database", "T01", 2, 3, 5),
    ("T04", "电影与评分数据准备", "Data prep", "T01", 2, 4, 7),
    ("T05", "后端基础框架与DAO开发", "Backend", "T02;T03", 4, 6, 10),
    ("T06", "用户与管理员模块开发", "User/admin", "T05", 4, 6, 10),
    ("T07", "电影与内容管理模块开发", "Movie/content", "T05", 5, 7, 12),
    ("T08", "用户互动与社区模块开发", "Interaction", "T06;T07", 4, 6, 11),
    ("T09", "在线推荐功能开发与接入", "Online rec", "T06;T07;T08", 5, 8, 14),
    ("T10", "独立混合推荐实验与权重调优", "Offline exp", "T04", 5, 8, 14),
    ("T11", "前端集成与系统测试", "Integration", "T02;T06;T07;T08;T09", 6, 9, 15),
    ("T12", "部署、验收与交付", "Release", "T10;T11", 2, 3, 6),
]


def prepare_project(root):
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    path = root / "config" / "wbs_tasks.csv"

    if not path.exists():
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "task_name", "short_name",
                "predecessors", "a", "m", "b"
            ])
            writer.writerows(TASKS_DEFAULT)

    tasks = []
    with path.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tasks.append({
                "id": r["task_id"],
                "name": r["task_name"],
                "short": r["short_name"],
                "pred": [
                    x for x in r["predecessors"].split(";") if x
                ],
                "a": float(r["a"]),
                "m": float(r["m"]),
                "b": float(r["b"]),
            })

    return tasks


# ==================================================
# 2. 任务依赖网络校验
# ==================================================

def topological_sort(tasks):
    ids = {t["id"] for t in tasks}

    for t in tasks:
        if not (0 < t["a"] <= t["m"] <= t["b"]):
            raise ValueError(f"{t['id']} 的PERT参数不合法")

        for p in t["pred"]:
            if p not in ids:
                raise ValueError(f"前置任务不存在: {p}")

    indegree = {t["id"]: len(t["pred"]) for t in tasks}
    successors = {t["id"]: [] for t in tasks}

    for t in tasks:
        for p in t["pred"]:
            successors[p].append(t["id"])

    queue = [
        t["id"] for t in tasks if indegree[t["id"]] == 0
    ]

    order = []

    while queue:
        u = queue.pop(0)
        order.append(u)

        for v in successors[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(order) != len(tasks):
        raise ValueError("任务依赖网络存在循环")

    return order, successors


# ==================================================
# 3. 可复现随机数与 Beta-PERT 采样
# ==================================================

class BetaRandom:
    """使用固定种子的 xorshift32 和 Gamma 比值生成 Beta 样本。"""

    def __init__(self, seed):
        self.state = seed & 0xffffffff
        self.spare = None

    def random(self):
        x = self.state
        x ^= (x << 13) & 0xffffffff
        x &= 0xffffffff
        x ^= x >> 17
        x &= 0xffffffff
        x ^= (x << 5) & 0xffffffff
        x &= 0xffffffff
        self.state = x
        return x / 4294967296.0

    def normal(self):
        if self.spare is not None:
            x = self.spare
            self.spare = None
            return x

        while True:
            u = 2 * self.random() - 1
            v = 2 * self.random() - 1
            s = u*u + v*v

            if 0 < s < 1:
                break

        factor = math.sqrt(-2 * math.log(s) / s)
        self.spare = v * factor
        return u * factor

    def gamma(self, shape):
        if shape < 1:
            return self.gamma(shape + 1) * (
                max(self.random(), 1e-15) ** (1 / shape)
            )

        d = shape - 1/3
        c = 1 / math.sqrt(9*d)

        while True:
            x = self.normal()
            v0 = 1 + c*x

            if v0 <= 0:
                continue

            v = v0**3
            u = self.random()

            if u < 1 - 0.0331*x**4:
                return d*v

            if math.log(u) < (
                0.5*x*x + d*(1-v+math.log(v))
            ):
                return d*v

    def beta(self, alpha, beta):
        x = self.gamma(alpha)
        y = self.gamma(beta)
        return x / (x+y)


def sample_pert(task, rng):
    a, m, b = task["a"], task["m"], task["b"]

    if a == b:
        return a

    lam = 4.0
    alpha = 1 + lam*(m-a)/(b-a)
    beta = 1 + lam*(b-m)/(b-a)

    x = rng.beta(alpha, beta)

    return a + (b-a)*x


def pert_mean(task):
    return (
        task["a"] + 4*task["m"] + task["b"]
    ) / 6


# ==================================================
# 4. CPM 单点工期分析
# ==================================================

def calculate_cpm(tasks, order, successors):
    by_id = {t["id"]: t for t in tasks}

    es, ef = {}, {}
    duration = {}

    for t in tasks:
        duration[t["id"]] = pert_mean(t)

    # 前推
    for tid in order:
        preds = by_id[tid]["pred"]

        es[tid] = max(
            [ef[p] for p in preds], default=0
        )

        ef[tid] = es[tid] + duration[tid]

    project_duration = max(ef.values())

    ls, lf = {}, {}

    # 后推
    for tid in reversed(order):
        lf[tid] = min(
            [ls[s] for s in successors[tid]],
            default=project_duration
        )

        ls[tid] = lf[tid] - duration[tid]

    rows = []

    for tid in order:
        slack = ls[tid] - es[tid]

        rows.append({
            "task_id": tid,
            "duration": duration[tid],
            "ES": es[tid],
            "EF": ef[tid],
            "LS": ls[tid],
            "LF": lf[tid],
            "total_float": max(0, slack),
            "critical": abs(slack) < 1e-7
        })

    # 回溯代表性关键路径
    end = max(ef, key=ef.get)
    path = [end]
    cur = end

    while by_id[cur]["pred"]:
        pred = max(
            by_id[cur]["pred"],
            key=lambda p: ef[p]
        )

        if abs(ef[pred] - es[cur]) > 1e-7:
            break

        path.append(pred)
        cur = pred

    path.reverse()

    return project_duration, path, rows


# ==================================================
# 5. 蒙特卡洛模拟
# ==================================================

def simulate(tasks, order, successors, runs, seed):
    if runs < 10000:
        raise ValueError("模拟次数必须至少为10000")

    index = {
        t["id"]: i for i, t in enumerate(tasks)
    }

    by_id = {t["id"]: t for t in tasks}

    n = len(tasks)

    rng = BetaRandom(seed)

    samples = np.zeros((runs, n))

    # 为每个任务随机采样工期
    for r in range(runs):
        for j, t in enumerate(tasks):
            samples[r, j] = sample_pert(t, rng)

    es = np.zeros_like(samples)
    ef = np.zeros_like(samples)

    # 每轮模拟执行前推
    for tid in order:
        j = index[tid]

        pred_idx = [
            index[p] for p in by_id[tid]["pred"]
        ]

        if pred_idx:
            es[:, j] = np.max(
                ef[:, pred_idx], axis=1
            )

        ef[:, j] = es[:, j] + samples[:, j]

    project_duration = np.max(ef, axis=1)

    # 后推，计算每轮模拟中的总时差
    ls = np.zeros_like(samples)
    lf = np.zeros_like(samples)

    for tid in reversed(order):
        j = index[tid]

        succ_idx = [
            index[s] for s in successors[tid]
        ]

        if succ_idx:
            lf[:, j] = np.min(
                ls[:, succ_idx], axis=1
            )
        else:
            lf[:, j] = project_duration

        ls[:, j] = lf[:, j] - samples[:, j]

    slack = ls - es
    critical = slack <= 1e-7

    # Spearman 秩相关
    ranks = np.argsort(
        np.argsort(samples, axis=0), axis=0
    ).astype(float)

    target_rank = np.argsort(
        np.argsort(project_duration)
    ).astype(float)

    target_rank -= target_rank.mean()
    target_norm = np.sqrt(np.sum(target_rank**2))

    sensitivity = []

    for j, t in enumerate(tasks):
        x = ranks[:, j] - ranks[:, j].mean()

        rho = np.sum(x*target_rank) / (
            np.sqrt(np.sum(x*x))*target_norm
        )

        sensitivity.append({
            "task_id": t["id"],
            "task_name": t["name"],
            "short_name": t["short"],
            "pert_mean": pert_mean(t),
            "spearman_rho": float(rho),
            "criticality_index": float(
                critical[:, j].mean()
            ),
            "sample_std": float(samples[:, j].std(ddof=1))
        })

    return project_duration, sensitivity


# ==================================================
# 6. 统计结果
# ==================================================

def get_summary(durations, cpm, deadline, runs, seed):
    ordered = np.sort(durations)

    def quantile(q):
        return float(ordered[int(q*(len(ordered)-1))])

    return {
        "runs": runs,
        "seed": seed,
        "deadline": deadline,
        "cpm_duration": cpm,
        "mc_mean": float(np.mean(durations)),
        "mc_std": float(np.std(durations, ddof=1)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p80": quantile(0.80),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "probability_on_time": float(
            np.mean(durations <= deadline)
        )
    }


# ==================================================
# 7. 结果保存
# ==================================================

def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(out, tasks, durations, summary, cpm_rows, sensitivity):
    out.mkdir(parents=True, exist_ok=True)

    save_csv(
        out / "simulation_durations.csv",
        [
            {"simulation_id": i+1, "duration": float(x)}
            for i, x in enumerate(durations)
        ],
        ["simulation_id", "duration"]
    )

    save_csv(
        out / "cpm_schedule.csv",
        cpm_rows,
        ["task_id", "duration", "ES", "EF",
         "LS", "LF", "total_float", "critical"]
    )

    save_csv(
        out / "task_sensitivity.csv",
        sensitivity,
        ["task_id", "task_name", "short_name",
         "pert_mean", "sample_std",
         "spearman_rho", "criticality_index"]
    )

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ==================================================
# 8. 可视化
# ==================================================

def draw_network(tasks, path):
    pos = {
        "T01": (0.0, 0.2),
        "T02": (2.2, 1.35),
        "T03": (2.2, 0.1),
        "T04": (2.2, -1.35),
        "T05": (4.5, 0.8),
        "T06": (6.8, 1.45),
        "T07": (6.8, 0.15),
        "T08": (9.1, 0.8),
        "T09": (11.3, 0.8),
        "T10": (8.0, -1.35),
        "T11": (13.5, 0.8),
        "T12": (15.7, 0.0)
    }

    critical_ids = {"T01", "T02", "T05", "T07", "T08", "T09", "T11", "T12"}
    critical_edges = {
        ("T01", "T02"), ("T02", "T05"), ("T05", "T07"),
        ("T07", "T08"), ("T08", "T09"), ("T09", "T11"), ("T11", "T12")
    }

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(-1.2, 16.9)
    ax.set_ylim(-2.15, 2.25)

    for t in tasks:
        x1, y1 = pos[t["id"]]
        for p in t["pred"]:
            x0, y0 = pos[p]
            is_crit_edge = (p, t["id"]) in critical_edges
            rad = 0.0
            if (p, t["id"]) in {("T02", "T11")}:
                rad = -0.18
            elif (p, t["id"]) in {("T06", "T09"), ("T07", "T09")}:
                rad = 0.12 if p == "T07" else -0.12
            elif (p, t["id"]) in {("T06", "T11"), ("T07", "T11"), ("T08", "T11")}:
                rad = -0.14 if p == "T06" else (0.18 if p == "T07" else -0.14)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.55,head_width=0.32",
                    color="#C0392B" if is_crit_edge else "#7F8C8D",
                    lw=2.0 if is_crit_edge else 1.1,
                    alpha=0.95 if is_crit_edge else 0.72,
                    shrinkA=30,
                    shrinkB=30,
                    connectionstyle=f"arc3,rad={rad}"
                ),
                zorder=1
            )

    for t in tasks:
        x, y = pos[t["id"]]
        is_crit = t["id"] in critical_ids
        ax.text(
            x, y,
            f'{t["id"]}\n{t["short"]}\n({t["a"]:.0f}/{t["m"]:.0f}/{t["b"]:.0f}d)',
            ha="center", va="center",
            fontsize=8.8,
            fontweight="bold" if is_crit else "normal",
            color="#1A252F",
            bbox=dict(
                boxstyle="round,pad=0.45",
                fc="#FDEBD0" if is_crit else "#EBF5FB",
                ec="#C0392B" if is_crit else "#2E86C1",
                lw=1.8 if is_crit else 1.2
            ),
            zorder=3
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#FDEBD0", edgecolor="#C0392B", linewidth=1.6, label="CPM Critical Task (a/m/b)"),
        Patch(facecolor="#EBF5FB", edgecolor="#2E86C1", linewidth=1.2, label="Non-Critical Task (a/m/b)")
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_title("WBS Task Dependency Network (MovieRecommendationSystem)", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def draw_histogram(durations, summary, path):
    plt.figure(figsize=(10, 5))

    plt.hist(
        durations, bins=45,
        density=True, edgecolor="black"
    )

    plt.axvline(
        summary["mc_mean"], linestyle="--",
        label=f'Mean={summary["mc_mean"]:.2f}'
    )

    plt.axvline(
        summary["p80"], linestyle="-.",
        label=f'P80={summary["p80"]:.2f}'
    )

    plt.axvline(
        summary["deadline"], linestyle=":",
        label=f'Deadline={summary["deadline"]:.1f}'
    )

    plt.xlabel("Project duration (working days)")
    plt.ylabel("Probability density")
    plt.title("Monte Carlo Project Duration Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def draw_sensitivity(data, path):
    rows = sorted(
        data, key=lambda x: x["spearman_rho"]
    )

    labels = [
        x["task_id"] + " " + x["short_name"]
        for x in rows
    ]

    values = [x["spearman_rho"] for x in rows]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Spearman Rank Correlation")
    plt.title("Project Duration Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def draw_criticality(data, path):
    rows = sorted(
        data, key=lambda x: x["criticality_index"]
    )

    labels = [
        x["task_id"] + " " + x["short_name"]
        for x in rows
    ]

    values = [
        x["criticality_index"]*100 for x in rows
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel("Criticality Index (%)")
    plt.title("Task Criticality Index")
    plt.xlim(0, 105)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# ==================================================
# 9. 自动生成 PDF 研究报告
# ==================================================

def build_pdf(root, tasks, summary, cpm_rows, sensitivity):
    out = root / "results"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    body = ParagraphStyle(
        "Body", fontName="STSong-Light",
        fontSize=9, leading=16, spaceAfter=6
    )

    title = ParagraphStyle(
        "Title", fontName="STSong-Light",
        fontSize=21, leading=30, alignment=1
    )

    h1 = ParagraphStyle(
        "H1", fontName="STSong-Light",
        fontSize=14, leading=22,
        spaceBefore=10, spaceAfter=7
    )

    small = ParagraphStyle(
        "Small", fontName="STSong-Light",
        fontSize=7, leading=10
    )

    def P(text, style=body):
        return Paragraph(escape(str(text)), style)

    story = []

    story.append(Spacer(1, 3*cm))
    story.append(P("软件工程管理课程作业", title))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph(
        "基于蒙特卡洛模拟的<br/>"
        "电影推荐系统开发项目工期风险评估",
        title
    ))

    story.append(Spacer(1, 1*cm))
    story.append(P("题目：基于蒙特卡洛模拟的软件项目工期风险评估"))
    story.append(P("案例：MovieRecommendationSystem"))
    story.append(Spacer(1, 2*cm))

    story.append(P(
        f'模拟次数：{summary["runs"]:,} 次；'
        f'随机种子：{summary["seed"]}；'
        f'截止日期：{summary["deadline"]:.0f} 个工作日'
    ))

    story.append(P(
        "说明：PERT 三点工期为规划估算值，不是历史工时实测数据。"
    ))

    story.append(PageBreak())

    def H(text):
        story.append(Paragraph(text, h1))

    def add_table(data, widths):
        tb = Table(data, colWidths=widths, repeatRows=1)
        tb.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), "STSong-Light"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(tb)
        story.append(Spacer(1, 8))

    H("摘要")
    story.append(P(
        f'本文以 MovieRecommendationSystem 为案例，'
        f'将项目拆分为 {len(tasks)} 个工作包，建立任务依赖网络，'
        f'采用 PERT 三点估算与 Beta-PERT 分布执行 '
        f'{summary["runs"]:,} 次蒙特卡洛模拟。'
        f'CPM 单点工期为 {summary["cpm_duration"]:.2f} 天，'
        f'模拟平均工期为 {summary["mc_mean"]:.2f} 天，'
        f'标准差为 {summary["mc_std"]:.2f} 天。'
        f'P50、P80、P90 分别为 '
        f'{summary["p50"]:.2f}、{summary["p80"]:.2f}、'
        f'{summary["p90"]:.2f} 天。'
        f'在 {summary["deadline"]:.0f} 天截止情景下，'
        f'按期完成概率为 {summary["probability_on_time"]:.2%}。'
        f'研究进一步结合 Spearman 秩相关与关键性指数识别高风险任务，'
        f'并提出进度管控措施。'
    ))

    story.append(P(
        "关键词：软件项目管理；WBS；PERT；Beta分布；"
        "蒙特卡洛模拟；关键路径；进度风险"
    ))

    H("1 绪论")
    story.append(P(
        "软件项目开发受到需求变化、模块耦合、数据质量、算法调试、"
        "接口联调和缺陷返工等因素影响，任务工期存在不确定性。"
        "传统 CPM 使用单点工期，能够计算基准项目工期和关键路径，"
        "但无法直接表达给定截止日期下的按期交付概率。"
        "本文通过 PERT 三点估算与蒙特卡洛模拟量化项目工期风险。"
    ))

    story.append(P(
        "本研究选择现有 MovieRecommendationSystem 作为案例。"
        "项目采用 Java、Spring Boot、MyBatis 和 MySQL，"
        "包含用户管理、电影内容、互动社区、在线推荐等模块。"
        "同时存在基于 CSV 的独立推荐实验，因此将在线推荐和离线实验"
        "分别纳入 WBS。"
    ))

    H("2 相关理论")
    story.append(P(
        "PERT 三点估算使用乐观工期 a、最可能工期 m 和悲观工期 b。"
        "其加权期望工期为 E(D)=(a+4m+b)/6。"
    ))

    story.append(P(
        "本文采用 Beta-PERT 分布：D=a+(b-a)X，"
        "其中 X 服从 Beta(alpha,beta)。"
        "alpha=1+4(m-a)/(b-a)，"
        "beta=1+4(b-m)/(b-a)。"
        "形状参数 lambda 取 4。"
    ))

    story.append(P(
        "每轮模拟按照任务依赖关系执行 CPM 前推，"
        "计算 ES 和 EF；项目工期为所有终止任务最早完成时间的最大值。"
        "后推计算 LS、LF 和总时差。重复模拟后，以样本均值、"
        "分位数和经验频率估计项目进度风险。"
    ))

    story.append(P(
        "敏感度分析采用 Spearman 秩相关系数，"
        "并计算关键性指数，即任务在模拟中处于关键路径的频率。"
    ))

    H("3 案例与 WBS")
    story.append(P(
        "项目范围包括需求分析、系统设计、数据准备、后端框架、"
        "用户管理、电影内容、用户互动、在线推荐、独立推荐实验、"
        "前端集成测试和部署交付。在线推荐路径使用数据库 DAO，"
        "独立推荐实验路径读取 movies.csv 和 ratings.csv。"
    ))

    wbs = [[
        P("编号", small), P("工作包", small),
        P("前置任务", small), P("a", small),
        P("m", small), P("b", small)
    ]]

    for t in tasks:
        wbs.append([
            P(t["id"], small), P(t["name"], small),
            P("、".join(t["pred"]) or "无", small),
            str(t["a"]), str(t["m"]), str(t["b"])
        ])

    add_table(wbs, [
        1.4*cm, 5.2*cm, 3.2*cm,
        1.0*cm, 1.0*cm, 1.0*cm
    ])

    story.append(P(
        "表1 项目 WBS 与 PERT 三点工期。"
        "以上工期为规划估算，不是历史工时记录。"
    ))

    story.append(Image(
        str(out / "task_network.png"),
        width=17*cm, height=5.9*cm
    ))

    H("4 仿真模型设计")
    story.append(P(
        f'模拟次数为 {summary["runs"]:,}，'
        f'随机种子为 {summary["seed"]}。'
        f'情景截止日期为 {summary["deadline"]:.0f} 个工作日。'
        f'所有任务采用完成—开始关系，工期单位为工作日。'
    ))

    story.append(P(
        "模型假设各任务工期独立，不考虑人员资源冲突、"
        "节假日、提前量、滞后量及任务间相关性。"
        "截止日期是用于概率分析的情景参数，并非实际合同日期。"
    ))

    H("5 仿真实验结果")

    story.append(Image(
        str(out / "duration_histogram.png"),
        width=16.5*cm, height=8.5*cm
    ))

    H("5.1 工期统计")
    stats = [
        ["指标", "结果"],
        ["CPM 单点工期", f'{summary["cpm_duration"]:.2f} 天'],
        ["模拟均值", f'{summary["mc_mean"]:.2f} 天'],
        ["标准差", f'{summary["mc_std"]:.2f} 天'],
        ["P50", f'{summary["p50"]:.2f} 天'],
        ["P80", f'{summary["p80"]:.2f} 天'],
        ["P90", f'{summary["p90"]:.2f} 天'],
        ["P95", f'{summary["p95"]:.2f} 天'],
        ["按期概率", f'{summary["probability_on_time"]:.2%}'],
    ]
    add_table(stats, [6*cm, 6*cm])

    story.append(P(
        "CPM 是使用各任务 PERT 均值计算的确定性基准。"
        "蒙特卡洛结果则反映多条路径随机变化后的工期分布。"
        "由于项目工期是多条路径完成时间的最大值，"
        "最大值运算具有非线性，因此两者的工期均值不必完全一致。"
    ))

    H("5.2 敏感度分析")

    story.append(Image(
        str(out / "sensitivity_spearman.png"),
        width=16.5*cm, height=9*cm
    ))

    story.append(Image(
        str(out / "criticality_index.png"),
        width=16.5*cm, height=9*cm
    ))

    top = sorted(
        sensitivity,
        key=lambda x: x["spearman_rho"],
        reverse=True
    )[:6]

    sens_table = [[
        "任务", "Spearman ρ", "关键性指数"
    ]]

    for x in top:
        sens_table.append([
            P(x["task_id"]+" "+x["task_name"], small),
            f'{x["spearman_rho"]:.3f}',
            f'{x["criticality_index"]:.1%}'
        ])

    add_table(sens_table, [
        7*cm, 3.5*cm, 3.5*cm
    ])

    H("6 风险分析与管控建议")

    story.append(P(
        "第一，在线推荐功能 T09 依赖用户、电影、评论数据和后端接口。"
        "建议提前冻结接口契约，准备冷启动、无评论用户和空推荐结果测试，"
        "并将缓存和异常回退纳入验收。"
    ))

    story.append(P(
        "第二，前端集成与系统测试 T11 容易集中暴露接口字段、"
        "会话状态和环境配置问题。建议持续集成、接口 Mock、"
        "冒烟测试前移，并在业务模块开发阶段同步联调。"
    ))

    story.append(P(
        "第三，电影内容模块 T07 和用户互动模块 T08 存在数据库字段、"
        "用户身份和业务状态依赖。建议先冻结数据模型，"
        "采用分模块验收和回归测试。"
    ))

    story.append(P(
        "第四，独立推荐实验 T10 应与网站在线推荐分开验收。"
        "建议检查 CSV 解析、训练测试数据隔离、评价指标定义和实验随机种子，"
        "保存原始结果以保证可复现性。"
    ))

    story.append(P(
        "进度缓冲方面，可使用 P80 作为较稳健的计划基准，"
        "P90 作为更高置信度的交付参考。若必须承诺 53 天，"
        "应优先保障敏感度排名靠前的任务，并在集成测试前预留缓冲。"
    ))

    H("7 结论与局限性")

    story.append(P(
        "本文构建了 12 个工作包的项目任务网络，"
        "使用 Beta-PERT 分布和 10,000 次蒙特卡洛模拟，"
        "获得项目工期分布、按期完成概率和任务敏感度。"
        "结果表明，单点 CPM 基准无法替代概率风险分析，"
        "Spearman 秩相关与关键性指数可用于识别高风险工作包。"
    ))

    story.append(P(
        "研究局限包括：PERT 工期来自规划估算而非历史实测；"
        "任务工期独立假设未覆盖共同原因风险；"
        "未考虑人员资源上限、节假日、返工循环和动态范围变更。"
        "后续可利用真实开发日志、缺陷记录和迭代数据校准参数。"
    ))

    H("参考文献")

    references = [
        "[1] Malcolm D G, Roseboom J H, Clark C E, Fazar W. "
        "Application of a Technique for Research and Development Program Evaluation. "
        "Operations Research, 1959, 7(5):646-669.",

        "[2] Clark C E. The PERT Model for the Distribution of an Activity Time. "
        "Operations Research, 1962, 10(3):405-406.",

        "[3] Project Management Institute. Practice Standard for Scheduling. "
        "3rd Edition. PMI, 2019.",

        "[4] Vose D. Risk Analysis: A Quantitative Guide. "
        "3rd Edition. Wiley, 2008.",

        "[5] Hulett D. Practical Schedule Risk Analysis. "
        "Gower Publishing, 2009.",

        "[6] 软件工程管理课程组. 软件工程管理课程作业2026. 2026.",

        "[7] 软件工程管理课程组. 软件项目进度管理：蒙特卡洛模拟与RCPSP. 2026."
    ]

    for r in references:
        story.append(P(r))

    H("附录：代码运行说明")
    story.append(P(
        "安装依赖：pip install -r requirements.txt。"
        "运行 python main.py --runs 10000 --seed 20261120 --deadline 53。"
        "所有实验数据、统计结果、图表和本 PDF 报告均自动生成在 results 目录。"
    ))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("STSong-Light", 8)
        canvas.drawString(
            1.5*cm, 0.8*cm,
            "MovieRecommendationSystem 项目进度风险评估"
        )
        canvas.drawRightString(
            A4[0]-1.5*cm, 0.8*cm,
            f"第 {doc.page} 页"
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out / "report.pdf"),
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.6*cm,
        bottomMargin=1.5*cm
    )

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer
    )


# ==================================================
# 10. 主程序
# ==================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--deadline", type=float, default=53)

    args = parser.parse_args()

    if args.runs < 10000:
        raise ValueError("课程作业要求至少模拟10000次")

    root = Path(__file__).resolve().parent
    tasks = prepare_project(root)

    out = root / "results"

    order, successors = topological_sort(tasks)

    cpm, critical_path, cpm_rows = calculate_cpm(
        tasks, order, successors
    )

    durations, sensitivity = simulate(
        tasks, order, successors,
        args.runs, args.seed
    )

    summary = get_summary(
        durations, cpm,
        args.deadline, args.runs, args.seed
    )

    save_results(
        out, tasks, durations,
        summary, cpm_rows, sensitivity
    )

    draw_network(tasks, out / "task_network.png")

    draw_histogram(
        durations, summary,
        out / "duration_histogram.png"
    )

    draw_sensitivity(
        sensitivity,
        out / "sensitivity_spearman.png"
    )

    draw_criticality(
        sensitivity,
        out / "criticality_index.png"
    )

    build_pdf(
        root, tasks, summary,
        cpm_rows, sensitivity
    )

    print("\n========== 实验完成 ==========")
    print(f"模拟次数：{args.runs}")
    print(f"CPM工期：{cpm:.2f} 天")
    print(f"模拟均值：{summary['mc_mean']:.2f} 天")
    print(f"标准差：{summary['mc_std']:.2f} 天")
    print(f"P50：{summary['p50']:.2f} 天")
    print(f"P80：{summary['p80']:.2f} 天")
    print(f"P90：{summary['p90']:.2f} 天")
    print(f"截止日期：{args.deadline} 天")
    print(f"按期概率：{summary['probability_on_time']:.2%}")
    print("CPM关键路径：", " -> ".join(critical_path))
    print("PDF报告：results/report.pdf")
    print("全部结果：results/")

    print("\n敏感度排名：")
    for x in sorted(
        sensitivity,
        key=lambda x: x["spearman_rho"],
        reverse=True
    )[:5]:
        print(
            x["task_id"],
            x["task_name"],
            f"rho={x['spearman_rho']:.3f}",
            f"criticality={x['criticality_index']:.1%}"
        )


if __name__ == "__main__":
    main()
