"""Generate the short T4 validation notebook without changing the full training notebook."""
from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

from make_notebook import cells


def main():
    quick = copy.deepcopy(cells)

    def replace(index, source):
        quick[index]["source"] = (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)

    replace(0, """
        # Q3 · T4 첫 검증 · 30분 목표

        기존과 같은 영상 전용 모델을 학습 16개·검증 4개 영상으로 짧게 검사합니다.
        실제 행동 정답은 읽지 않습니다. 과거 32장 → 미래 32장 예측과 copy-last를 비교합니다.
        최대 300회 업데이트, 학습 시간 상한 10분입니다. 설정 실행부터 20분을 예산으로 두고
        마지막 검증·저장에 2분을 남깁니다. GPU 할당·로그인 지연은 별도로 확인해야 합니다.
    """)
    replace(1, """
        ## 1. T4 검증 설정
        약 20MB의 선택 영상 묶음을 사용합니다. GPU 런타임에서 셀을 순서대로 실행하세요.
        설정 셀을 실행하는 순간 시간 예산을 시작합니다. 이 노트북은 짧은 학습을 실행합니다.
        검증 4개에 대한 결과이므로 전체 데이터의 성능을 대표하지는 않습니다.
    """)
    replace(2, '''
        import time
        REPO_URL = "https://github.com/seungjoolee24/krafton-q3-video-worldmodel.git"
        GIT_REF = "main"
        PROFILE = "t4_quick"
        RUN_NAME = "video-only-t4-first-v1"
        DRIVE_ROOT = "/content/drive/MyDrive/krafton-q3-video-worldmodel"
        DATA_ZIP = DRIVE_ROOT + "/track3-kit-t4-quick.zip"
        DATA_SHA256 = "c9dc3afc3691245366de08f8c0fd6665fd5fb410bb136999b2147cd717955882"
        RESUME_CHECKPOINT = ""
        MAX_STEPS = 300
        BATCH_SIZE = 4
        TRAIN_BUDGET_SECONDS = 600
        TOTAL_BUDGET_MINUTES = 20
        QUICK_STARTED = time.time()
        QUICK_DEADLINE = QUICK_STARTED + TOTAL_BUDGET_MINUTES * 60
        print("Validation deadline:", time.strftime("%H:%M:%S UTC", time.gmtime(QUICK_DEADLINE)))
    ''')
    quick[2]["metadata"] = {}
    environment = "".join(quick[4]["source"]).replace('{"pilot", "full"}', '{"t4_quick"}')
    replace(4, environment)
    replace(9, """
        ## 5. 선택 영상 특징 캐시
        원본 MP4 20개를 고정 인코더로 변환합니다. 캐시는 약 0.35GB입니다.
        실제 액션 파일은 데이터 묶음에 포함되어 있지 않습니다.
    """)
    replace(11, """
        ## 6. 짧은 GPU 학습과 검증
        학습 전 기준선을 먼저 저장하고, 최대 300회 또는 학습 시간 상한까지 업데이트합니다.
        예측 길이는 1 → 4 → 8 → 16 → 32로 늘립니다. 종료 시점의 모델도 검증하고 저장합니다.
        시간 상한은 업데이트 사이에서 적용하며 마지막 검증·저장 시간이 추가됩니다.
    """)
    training = "".join(quick[12]["source"])
    training = '''remaining = QUICK_DEADLINE - time.time() - 120
assert remaining > 0, "Setup exhausted the time budget; check GPU/Drive connectivity."
training_budget = min(TRAIN_BUDGET_SECONDS, remaining)
print({"training_seconds_limit": training_budget, "max_updates": MAX_STEPS})
''' + training
    training = training.replace('subprocess.run(command, check=True)',
                                'command.extend(["--max-seconds", str(training_budget)])\nsubprocess.run(command, check=True)')
    replace(12, training)
    replace(13, """
        ## 7. 첫 검증 결과
        학습 전후 RGB MSE, copy-last, 최종 손실과 비교 영상을 확인합니다.
        아래 영상은 마지막 완료된 업데이트의 모델입니다. 최저 오차 모델은 별도의 best.pt입니다.
        개선 여부는 결과로 판단합니다. 수치가 낮아도 실제 힘 복원이 검증된 것은 아닙니다.
    """)
    replace(14, '''
        from IPython.display import display, Video
        RESULT_ROOT = LOCAL_RUN if (LOCAL_RUN / "completion.json").exists() else DRIVE_RUN
        completion = json.loads((RESULT_ROOT / "completion.json").read_text())
        REPORT = LOCAL_RUN / "validation" / f"step_{completion['step']:06d}"
        if not (REPORT / "metrics.json").exists():
            REPORT = DRIVE_RUN / "validation" / f"step_{completion['step']:06d}"
        metrics = json.loads((REPORT / "metrics.json").read_text())
        initial_path = RESULT_ROOT / "validation" / "step_000000" / "metrics.json"
        initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
        loss_path = RESULT_ROOT / "training.jsonl"
        logs = [json.loads(line) for line in loss_path.read_text().splitlines()] if loss_path.exists() else []
        display({"completed_updates": completion["step"], "stop_reason": completion["reason"],
                 "total_elapsed_minutes": (time.time() - QUICK_STARTED) / 60,
                 "initial_rgb_mse": initial["mse"]["mean"] if initial else None,
                 "final_rgb_mse": metrics["mse"]["mean"],
                 "copy_last_rgb_mse": metrics["copy_last_mse"]["mean"],
                 "final_horizon_errors": metrics["mse"],
                 "last_training_log": logs[-1] if logs else None,
                 "gpu_peak_memory_gb": completion["gpu_peak_memory_gb"],
                 "actions_used": False, "saved_results": str(DRIVE_RUN)})
        for video in sorted(REPORT.glob("*.mp4")):
            display(Video(str(video), embed=True, width=768))
        print("Results saved:", DRIVE_RUN)
    ''')
    for index, cell in enumerate(quick):
        cell["id"] = f"q3-t4-{index:02d}"
    path = Path(__file__).resolve().parents[1] / "notebooks" / "validate_t4_30min.ipynb"
    notebook = {"cells": quick, "metadata": {"accelerator": "GPU", "colab": {"name": path.name},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
