"""Live Gemini smoke test for checkpoint/final summaries (FR-07, FR-08).

Exercises the production summary path with a fixture lecture window: config ->
``create_summarizer`` -> prompt assembly -> Gemini. Only text is sent (the
transcript and slide blocks); no audio or screenshots. Use it to verify
``GEMINI_API_KEY``, the configured model id, and the prompt output before a full
lecture session::

    .\\.venv\\Scripts\\python.exe scripts\\smoke_summary.py
    .\\.venv\\Scripts\\python.exe scripts\\smoke_summary.py --model gemini-3.5-flash-lite

The manual check for phase 6 is whether the final markdown contains the four
sections and at least 8 of the fixture lecture's 10 key points (bulleted in
``FIXTURE_TRANSCRIPT`` below).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tpdk_overlay.config import load_config  # noqa: E402
from tpdk_overlay.ports import (  # noqa: E402
    CheckpointSummaryRequest,
    FinalSummaryRequest,
    SummarizerUnavailable,
)
from tpdk_overlay.summarize import create_summarizer  # noqa: E402

# Ten key points for the phase 6 acceptance check: gradient descent update,
# non-convex loss, overshoot with a fixed step, step decay, cosine annealing,
# warmup, batch size effect, momentum, adaptive optimizers, and validation loss.
_FIXTURE_SEGMENTS: list[tuple[int, int, str, str]] = [
    (
        0,
        5200,
        "Today we cover gradient descent and why the learning rate matters.",
        "Hari ini kita membahas gradient descent dan mengapa learning rate penting.",
    ),
    (
        5200,
        11000,
        "The loss surface is not convex in general, so a fixed step size can "
        "overshoot the minimum.",
        "Permukaan loss umumnya tidak konveks, jadi ukuran langkah tetap bisa "
        "melewati minimum.",
    ),
    (
        11000,
        17500,
        "Step decay multiplies the learning rate by a factor every few epochs, "
        "for example 0.1 every 30 epochs.",
        "Step decay mengalikan learning rate dengan faktor tertentu setiap "
        "beberapa epoch, misalnya 0.1 tiap 30 epoch.",
    ),
    (
        17500,
        23000,
        "Cosine annealing follows a cosine curve instead of a step function, "
        "which avoids a sudden drop in progress.",
        "Cosine annealing mengikuti kurva kosinus, bukan fungsi tangga, sehingga "
        "menghindari penurunan progres yang mendadak.",
    ),
    (
        23000,
        28500,
        "Warmup starts with a tiny learning rate and ramps up over the first few "
        "hundred steps to stabilize early training.",
        "Warmup memulai dengan learning rate sangat kecil lalu menaikkannya dalam "
        "beberapa ratus langkah pertama untuk menstabilkan awal pelatihan.",
    ),
    (
        28500,
        34000,
        "A larger batch size reduces gradient noise but may need a larger "
        "learning rate or more epochs to converge.",
        "Batch size yang lebih besar mengurangi noise gradien tetapi mungkin "
        "butuh learning rate lebih besar atau lebih banyak epoch.",
    ),
    (
        34000,
        39500,
        "Momentum accumulates past gradients so the update keeps moving through "
        "flat regions and small local minima.",
        "Momentum mengakumulasi gradien sebelumnya sehingga update tetap bergerak "
        "melewati daerah datar dan minimum lokal kecil.",
    ),
    (
        39500,
        45000,
        "Adaptive optimizers such as Adam scale the step per parameter, which "
        "helps when features have very different scales.",
        "Optimizer adaptif seperti Adam menskalakan langkah per parameter, "
        "berguna saat fitur punya skala yang sangat berbeda.",
    ),
    (
        45000,
        50500,
        "Always monitor validation loss, not only training loss, because a "
        "falling training loss can hide overfitting.",
        "Selalu pantau validation loss, bukan hanya training loss, karena "
        "training loss yang turun bisa menyembunyikan overfitting.",
    ),
    (
        50500,
        56000,
        "For the assignment, compare a constant learning rate with step decay on "
        "the same seed and report both curves.",
        "Untuk tugas, bandingkan learning rate konstan dengan step decay pada "
        "seed yang sama dan laporkan kedua kurvanya.",
    ),
]

FIXTURE_TRANSCRIPT = "\n".join(
    f"[{start}-{end} ms] {text} | {translation}"
    for start, end, text, translation in _FIXTURE_SEGMENTS
)

FIXTURE_SLIDES = """\
[9000 ms]
Gradient descent
theta <- theta - eta * grad L(theta)

[18000 ms]
Learning rate schedules
step decay: eta * 0.1 every 30 epochs
cosine annealing

[30000 ms]
Stability tools
warmup, momentum, Adam
"""


async def _run(args: argparse.Namespace) -> int:
    config = load_config()
    if args.model:
        config = config.model_copy(
            update={
                "summary": config.summary.model_copy(update={"model": args.model})
            }
        )
    summarizer = create_summarizer(config.summary)
    if summarizer is None:
        if not config.summary.enabled:
            reason = "summary.enabled is false in configs/default.yaml"
        else:
            reason = f"{config.summary.api_key_env} is not set (see .env)"
        print(f"cannot run: {reason}", file=sys.stderr)
        return 2

    source = config.app.language.source
    target = args.target or config.app.language.target
    print(f"model: {config.summary.model}  languages: {source} -> {target}")

    checkpoint = CheckpointSummaryRequest(
        checkpoint_id=1,
        t_from_ms=0,
        t_to_ms=60_000,
        transcript=FIXTURE_TRANSCRIPT.strip(),
        slides=FIXTURE_SLIDES.strip(),
        source_lang=source,
        target_lang=target,
    )
    print("requesting checkpoint summary ...")
    try:
        checkpoint_markdown = await summarizer.checkpoint(checkpoint)

        print("requesting final summary ...")
        final_markdown = await summarizer.final(
            FinalSummaryRequest(
                checkpoints=[checkpoint_markdown],
                transcript="",
                source_lang=source,
                target_lang=target,
            )
        )
    except SummarizerUnavailable as exc:
        print(f"summary failed: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / f"checkpoint_{stamp}.md"
    final_path = args.out_dir / f"final_{stamp}.md"
    checkpoint_path.write_text(checkpoint_markdown.rstrip() + "\n", encoding="utf-8")
    final_path.write_text(final_markdown.rstrip() + "\n", encoding="utf-8")

    print("\n=== checkpoint ===")
    print(checkpoint_markdown)
    print("\n=== final ===")
    print(final_markdown)
    print(f"\nwrote {checkpoint_path}")
    print(f"wrote {final_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override summary.model")
    parser.add_argument("--target", default=None, help="override the target language")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "summary_smoke",
        help="where the generated markdown lands",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
