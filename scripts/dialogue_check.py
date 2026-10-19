# 对话纪律检查：家规"主角单次说话不超过两句"
# 用法：python scripts/dialogue_check.py <语料目录> [--min-sentences 3]
# 输出：每个引号内句终符(。！？；)≥N 的发言，标注归属（段内含"林晚"）
import argparse
import re
from pathlib import Path

QUOTE_RE = re.compile(r"“([^”]{2,})”")
END_RE = re.compile(r"[。！？!?；;]")


def check_file(path: Path, min_sentences: int):
    hits = []
    para_idx = 0
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        para_idx += 1
        for m in QUOTE_RE.finditer(line):
            quote = m.group(1)
            n = len(END_RE.findall(quote))
            if n < min_sentences:
                continue
            speaker = (
                "林晚?"
                if ("林晚" in line or "她说" in line)
                else "他人/旁白"
            )
            hits.append(
                {
                    "line": lineno,
                    "sentences": n,
                    "speaker": speaker,
                    "head": quote[:38],
                }
            )
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--min-sentences", type=int, default=3)
    args = parser.parse_args()

    total = 0
    by_ch = {}
    for f in sorted(Path(args.directory).glob("ch*.txt")):
        hits = check_file(f, args.min_sentences)
        if hits:
            by_ch[f.stem] = hits
            total += len(hits)

    for ch, hits in by_ch.items():
        worst = max(h["sentences"] for h in hits)
        lin = sum(1 for h in hits if "?" in h["speaker"])
        print(
            f"{ch}: {len(hits)}处 (最重{worst}句, 疑似林晚{lin}处)"
        )
        for h in sorted(
            hits, key=lambda x: -x["sentences"]
        )[:3]:
            mark = "★" if "?" in h["speaker"] else " "
            print(
                f" {mark} L{h['line']} [{h['sentences']}句] "
                f"{h['head']}"
            )
    print(f"\n合计: {total} 处 (阈值≥{args.min_sentences}句)")


if __name__ == "__main__":
    main()
