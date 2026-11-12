"""Compare PLaMo-3-Translate GGUF outputs with NLLB on the fixed lecture set.

Reads a ``bench_mt_plamo.py`` JSON report for the PLaMo side, runs the project
default NLLB adapter on the same eight sentences, and scores both against the
human reference in ``bench_mt.py`` with chrF++ (sacrebleu, word_order=2).

    python scripts/compare_mt_quality.py --plamo-json data/benchmarks/<report>.json
    python scripts/compare_mt_quality.py --plamo-json <report> --nllb-device cuda \\
        --nllb-dtype float16 --nllb-provider nllb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import sacrebleu

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bench_mt import EN, ID  # noqa: E402

from tpdk_overlay.config import load_config  # noqa: E402
from tpdk_overlay.mt import WorkerTranslator, create_translator  # noqa: E402

_REFERENCES = {"en->id": ID, "id->en": EN}
_SOURCES = {"en->id": EN, "id->en": ID}


def _plamo_outputs(report: dict[str, Any]) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    for name in ("en->id", "id->en"):
        entry = report["directions"].get(name)
        if entry is None or len(entry["outputs"]) != len(_SOURCES[name]):
            raise ValueError(f"report is missing complete {name} outputs")
        outputs[name] = [str(text).strip() for text in entry["outputs"]]
    return outputs


def _nllb_outputs(args: argparse.Namespace) -> dict[str, list[str]]:
    config = load_config().mt
    updates: dict[str, Any] = {}
    if args.nllb_provider is not None:
        updates["provider"] = args.nllb_provider
    if args.nllb_device is not None:
        updates["device"] = args.nllb_device
    if args.nllb_dtype is not None:
        updates["dtype"] = args.nllb_dtype
    if updates:
        config = config.model_copy(update=updates)

    translator = cast(WorkerTranslator, create_translator(config))
    try:
        translator.warmup()
        return {
            name: translator.translate(
                list(_SOURCES[name]), source_lang=name[:2], target_lang=name[-2:]
            )
            for name in ("en->id", "id->en")
        }
    finally:
        translator.close()


def _chrf(hypotheses: list[str], references: list[str]) -> float:
    return sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score PLaMo vs NLLB with chrF++.")
    parser.add_argument("--plamo-json", type=Path, required=True)
    parser.add_argument("--nllb-provider", choices=("nllb", "nllb_ct2"), default="nllb_ct2")
    parser.add_argument("--nllb-device", default=None)
    parser.add_argument("--nllb-dtype", default=None)
    args = parser.parse_args(argv)

    report = json.loads(args.plamo_json.read_text(encoding="utf-8"))
    plamo = _plamo_outputs(report)
    nllb = _nllb_outputs(args)

    summary: dict[str, Any] = {"plamo_json": str(args.plamo_json), "directions": {}}
    for name in ("en->id", "id->en"):
        sources, references = _SOURCES[name], _REFERENCES[name]
        plamo_score = _chrf(plamo[name], references)
        nllb_score = _chrf(nllb[name], references)
        summary["directions"][name] = {
            "plamo_chrfpp": round(plamo_score, 3),
            "nllb_chrfpp": round(nllb_score, 3),
            "rows": [
                {
                    "source": source,
                    "reference": reference,
                    "plamo": plamo_text,
                    "nllb": nllb_text,
                }
                for source, reference, plamo_text, nllb_text in zip(
                    sources, references, plamo[name], nllb[name], strict=True
                )
            ],
        }
        print(f"\n== {name}  chrF++  PLaMo {plamo_score:.2f}  NLLB {nllb_score:.2f}")
        for row in summary["directions"][name]["rows"]:
            print(f"  SRC : {row['source']}")
            print(f"  REF : {row['reference']}")
            print(f"  PLM : {row['plamo']}")
            print(f"  NLB : {row['nllb']}")
            print()

    output = args.plamo_json.with_name(args.plamo_json.stem + "_quality.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
