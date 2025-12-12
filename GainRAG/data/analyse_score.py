# %%
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Tuple, Optional

DATA_PATH = Path(
    "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/"
    "GainRAG/GainRAG/data/without_pse_nq_train_selector_debug_data.jsonl"
)


def safe_mean(xs: List[float]) -> float:
    return float(mean(xs)) if xs else float("nan")


def safe_std(xs: List[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else float("nan")


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    """简单 Pearson 相关系数，长度不够或方差为 0 就返回 None"""
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = safe_mean(x), safe_mean(y)
    vx = [xi - mx for xi in x]
    vy = [yi - my for yi in y]
    denom_x = sum(v * v for v in vx)
    denom_y = sum(v * v for v in vy)
    if denom_x == 0 or denom_y == 0:
        return None
    num = sum(a * b for a, b in zip(vx, vy))
    return num / (denom_x ** 0.5 * denom_y ** 0.5)


def describe(name: str, xs: List[float]):
    if not xs:
        print(f"{name}: 无数据")
        return
    print(
        f"{name}:\n"
        f"  count = {len(xs)}\n"
        f"  min   = {min(xs):.6f}\n"
        f"  max   = {max(xs):.6f}\n"
        f"  mean  = {safe_mean(xs):.6f}\n"
        f"  std   = {safe_std(xs):.6f}"
    )


def main():
    if not DATA_PATH.is_file():
        print(f"文件不存在: {DATA_PATH}")
        return

    num_samples = 0

    # 全局累积
    all_ppl_original = []
    all_ppl_transformed = []
    all_ppl_softmax = []
    all_ppl_normalized = []

    all_ret_original = []
    all_ret_normalized = []
    all_combined = []

    # 用于相关性分析的扁平化向量
    corr_pplnorm_retnorm_x = []
    corr_pplnorm_retnorm_y = []

    corr_combined_pplnorm_x = []
    corr_combined_pplnorm_y = []

    corr_combined_retnorm_x = []
    corr_combined_retnorm_y = []

    # top1 一致性统计
    same_top_ppl_vs_ret = 0
    same_top_comb_vs_ret = 0
    same_top_comb_vs_ppl = 0
    valid_top_samples = 0  # 有完整三种分数的样本数

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            num_samples += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[警告] 第 {num_samples} 行 JSON 解析失败，跳过")
                continue

            dbg = obj.get("debug_info", {})
            ppl_original = dbg.get("ppl_original") or []
            ppl_transformed = dbg.get("ppl_transformed") or []
            ppl_softmax = dbg.get("ppl_softmax") or []
            ppl_normalized = dbg.get("ppl_normalized") or []

            ret_original = dbg.get("retrieval_original") or []
            ret_normalized = dbg.get("retrieval_normalized") or []
            combined = dbg.get("combined_scores") or []

            # 维度检查：我们希望长度一致（一般 = num_passages）
            lens = {
                "ppl_original": len(ppl_original),
                "ppl_transformed": len(ppl_transformed),
                "ppl_softmax": len(ppl_softmax),
                "ppl_normalized": len(ppl_normalized),
                "retrieval_original": len(ret_original),
                "retrieval_normalized": len(ret_normalized),
                "combined_scores": len(combined),
            }
            # 只在 debug 时提示一次长度不一致
            if len(set(lens.values()) - {0}) > 1:
                print(f"[警告] 第 {num_samples} 行各分数字段长度不一致: {lens}")

            # 全局分布统计
            all_ppl_original.extend(ppl_original)
            all_ppl_transformed.extend(ppl_transformed)
            all_ppl_softmax.extend(ppl_softmax)
            all_ppl_normalized.extend(ppl_normalized)

            all_ret_original.extend(ret_original)
            all_ret_normalized.extend(ret_normalized)
            all_combined.extend(combined)

            # 相关性分析（逐元素拼接）
            n = min(len(ppl_normalized), len(ret_normalized), len(combined))
            if n > 0:
                corr_pplnorm_retnorm_x.extend(ppl_normalized[:n])
                corr_pplnorm_retnorm_y.extend(ret_normalized[:n])

                corr_combined_pplnorm_x.extend(combined[:n])
                corr_combined_pplnorm_y.extend(ppl_normalized[:n])

                corr_combined_retnorm_x.extend(combined[:n])
                corr_combined_retnorm_y.extend(ret_normalized[:n])

            # top1 一致性
            if (
                len(ppl_normalized) == len(ret_normalized) == len(combined)
                and len(combined) > 0
            ):
                valid_top_samples += 1
                idx_ppl_top = max(range(len(ppl_normalized)), key=lambda i: ppl_normalized[i])
                idx_ret_top = max(range(len(ret_normalized)), key=lambda i: ret_normalized[i])
                idx_comb_top = max(range(len(combined)), key=lambda i: combined[i])

                if idx_ppl_top == idx_ret_top:
                    same_top_ppl_vs_ret += 1
                if idx_comb_top == idx_ret_top:
                    same_top_comb_vs_ret += 1
                if idx_comb_top == idx_ppl_top:
                    same_top_comb_vs_ppl += 1

    # ========== 打印统计结果 ==========
    print("====== 样本基本信息 ======")
    print(f"样本总数: {num_samples}")
    print(f"用于 top1 一致性分析的样本数: {valid_top_samples}")
    print()

    print("====== 分数整体分布 ======")
    describe("ppl_original", all_ppl_original)
    print()
    describe("ppl_transformed", all_ppl_transformed)
    print()
    describe("ppl_softmax", all_ppl_softmax)
    print()
    describe("ppl_normalized", all_ppl_normalized)
    print()
    describe("retrieval_original", all_ret_original)
    print()
    describe("retrieval_normalized", all_ret_normalized)
    print()
    describe("combined_scores", all_combined)
    print()

    print("====== 分数之间的相关性（Pearson）======")
    r1 = pearson(corr_pplnorm_retnorm_x, corr_pplnorm_retnorm_y)
    r2 = pearson(corr_combined_pplnorm_x, corr_combined_pplnorm_y)
    r3 = pearson(corr_combined_retnorm_x, corr_combined_retnorm_y)

    print(f"ppl_normalized vs retrieval_normalized: r = {r1:.4f}" if r1 is not None else
          "ppl_normalized vs retrieval_normalized: 无法计算（方差为 0 或数据不足）")
    print(f"combined_scores vs ppl_normalized:     r = {r2:.4f}" if r2 is not None else
          "combined_scores vs ppl_normalized: 无法计算")
    print(f"combined_scores vs retrieval_normalized: r = {r3:.4f}" if r3 is not None else
          "combined_scores vs retrieval_normalized: 无法计算")
    print()

    print("====== top1 一致性（基于 ppl_normalized / retrieval_normalized / combined_scores）======")
    if valid_top_samples > 0:
        print(
            f"ppl_top == retrieval_top 比例: "
            f"{same_top_ppl_vs_ret}/{valid_top_samples} "
            f"({same_top_ppl_vs_ret / valid_top_samples:.2%})"
        )
        print(
            f"combined_top == retrieval_top 比例: "
            f"{same_top_comb_vs_ret}/{valid_top_samples} "
            f"({same_top_comb_vs_ret / valid_top_samples:.2%})"
        )
        print(
            f"combined_top == ppl_top 比例: "
            f"{same_top_comb_vs_ppl}/{valid_top_samples} "
            f"({same_top_comb_vs_ppl / valid_top_samples:.2%})"
        )
    else:
        print("没有长度一致且非空的样本，无法计算 top1 一致性。")


if __name__ == "__main__":
    main()
# %%
