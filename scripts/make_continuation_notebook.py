"""Continue the existing pilot with the same recipe and optimizer state."""
from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

from make_notebook import cells


def main():
    continuation = copy.deepcopy(cells)

    def replace(index, source):
        continuation[index]["source"] = (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)

    replace(0, """
        # Q3 · T4 · 2,000 → 6,000번 추가 학습

        이전 모델의 **2,000번 체크포인트에서 추가 4,000번** 학습합니다.
        같은 학습 128개·검증 32개 영상, 같은 모델·손실·학습률을 사용합니다.
        모델·optimizer·AMP scaler·난수·step을 모두 이어받습니다.
        32프레임 연속 예측을 계속 학습하며, 500번마다 같은 검증 영상에서 비교합니다.
        이전 실험은 보존하고 새 폴더에 결과를 저장합니다. 실제 행동 라벨은 사용하지 않습니다.
    """)
    replace(1, """
        ## 1. 연속 학습 설정
        `MAX_STEPS=6000`은 이번 단계의 누적 종료 지점입니다.
        첫 실행은 이전 latest.pt에서 2,000 → 6,000번까지 학습합니다.
        같은 새 실행 폴더의 latest.pt가 있으면 그 시점부터 남은 업데이트를 재개합니다.
    """)
    replace(2, '''
        REPO_URL = "https://github.com/seungjoolee24/krafton-q3-video-worldmodel.git"
        GIT_REF = "main"
        PROFILE = "pilot"
        RUN_NAME = "video-only-t4-6000-v1"
        DRIVE_ROOT = "/content/drive/MyDrive/krafton-q3-video-worldmodel"
        SOURCE_RUN_NAME = "video-only-t4-expanded-v1"
        DATA_ZIP = DRIVE_ROOT + "/track3-kit.zip"
        DATA_SHA256 = "dbd9da58b386782533913da2c44b088a6c25c6119ba6efbe680dae56bf4422ee"
        MAX_STEPS = 6000
        BATCH_SIZE = 4
        print({"source_step": 2000, "stop_step": MAX_STEPS, "additional_updates": MAX_STEPS - 2000,
               "train_videos": 128, "dev_videos": 32, "training_horizon": 32, "actions_used": False})
    ''')
    continuation[2]["metadata"] = {}
    data = "".join(continuation[6]["source"])
    data = data.replace('drive.mount("/content/drive")', 'drive.mount("/content/drive", timeout_ms=60000)')
    data = data.replace('config["batch_size"] = BATCH_SIZE', '''config["batch_size"] = BATCH_SIZE
config["eval_every"] = 500
config["checkpoint_every"] = 250
config["log_every"] = 50
config["evaluate_at_start"] = True
SOURCE_RUN = DRIVE / "runs" / SOURCE_RUN_NAME''')
    replace(6, data)
    replace(7, """
        ## 4. GPU 동작 확인
        기존 학습 코드와 데이터 구조를 그대로 사용합니다.
        짧은 합성 GPU 동작 검사를 통과한 뒤 실제 체크포인트로 재개합니다.
    """)
    replace(8, '''
        subprocess.run([sys.executable, "-m", "video_wam.cli", "smoke",
                        "--kit", str(KIT), "--device", "cuda"], check=True)
    ''')
    replace(11, """
        ## 6. 2,000 → 6,000번 학습
        이전 실험의 최저 오차 모델과 2,000번 비교 결과를 새 폴더에도 보존합니다.
        이번 추가 학습은 시작부터 끝까지 32프레임 예측을 사용합니다.
        로그를 화면에 표시하고, 250번마다 저장하며 500번마다 32개 영상을 검증합니다.
        중단 버튼을 누르면 마지막 완료된 업데이트의 latest.pt를 저장합니다.
    """)
    replace(12, '''
        import signal
        from video_wam.utils import persist_file, atomic_json
        DRIVE_RUN.mkdir(parents=True, exist_ok=True)
        resume = DRIVE_RUN / "latest.pt"
        if not resume.is_file():
            resume = SOURCE_RUN / "latest.pt"
            assert resume.is_file(), f"Missing source checkpoint: {resume}"
            source = torch.load(resume, map_location="cpu", weights_only=True)
            assert source["step"] == 2000, "Expected the completed 2,000-step pilot."
            assert source["cache_fingerprint"] == index["fingerprint"], "Selected data must match the original run."
            assert source["model_config"] == config["model"]
            source_best = torch.load(SOURCE_RUN / "best.pt", map_location="cpu", weights_only=True)
            assert source_best["cache_fingerprint"] == index["fingerprint"]
            if not (DRIVE_RUN / "best.pt").exists():
                persist_file(SOURCE_RUN / "best.pt", DRIVE_RUN)
            for step in sorted({source["step"], source_best["step"]}):
                source_report = SOURCE_RUN / "validation" / f"step_{step:06d}"
                assert (source_report / "metrics.json").is_file()
                destination = DRIVE_RUN / "validation" / source_report.name
                for artifact in source_report.iterdir():
                    if artifact.is_file() and not (destination / artifact.name).exists():
                        persist_file(artifact, destination)
            lineage = {"source_run": str(SOURCE_RUN), "source_step": source["step"],
                       "source_git_revision": source["git_revision"], "target_step": MAX_STEPS,
                       "optimizer_restored": True, "rng_restored": True,
                       "cache_fingerprint": index["fingerprint"], "actions_used": False}
            atomic_json(LOCAL_RUN / "continuation.json", lineage)
            persist_file(LOCAL_RUN / "continuation.json", DRIVE_RUN)
            del source, source_best
        command = [sys.executable, "-u", "-m", "video_wam.cli", "train",
                   "--kit", str(KIT), "--cache", str(CACHE), "--config", str(CONFIG_PATH),
                   "--out", str(LOCAL_RUN), "--persist", str(DRIVE_RUN),
                   "--resume", str(resume), "--max-steps", str(MAX_STEPS), "--device", "cuda"]
        print("Resume full training state:", resume, flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            raise
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    ''')
    replace(13, """
        ## 7. 추가 학습의 효과
        같은 검증 영상 32개에서 추가 학습 전(2,000번)과 이후를 비교합니다.
        새 실험에서 개선되지 않으면 이전의 best.pt가 그대로 유지됩니다.
        물체 영역의 오차는 색 기반 보조 지표입니다. 비교 영상의 위치·각도·선명도도 함께 확인하세요.
        런타임 초기화 후에는 이 결과 셀만 실행해 Drive의 완료 결과를 읽을 수 있습니다.
    """)
    replace(14, '''
        import json
        import torch
        from pathlib import Path
        from google.colab import drive
        from IPython.display import display, HTML, Video
        drive.mount("/content/drive", timeout_ms=60000)
        if "DRIVE_RUN" not in globals():
            DRIVE_RUN = Path("/content/drive/MyDrive/krafton-q3-video-worldmodel/runs/video-only-t4-6000-v1")
            LOCAL_RUN = Path("/content/q3-runs/video-only-t4-6000-v1")
        completion_path = LOCAL_RUN / "completion.json"
        if not completion_path.exists():
            completion_path = DRIVE_RUN / "completion.json"
        completion = json.loads(completion_path.read_text())

        def report_for(step):
            relative = Path("validation") / f"step_{step:06d}"
            folder = LOCAL_RUN / relative
            if not (folder / "metrics.json").exists():
                folder = DRIVE_RUN / relative
            return folder, json.loads((folder / "metrics.json").read_text())

        _, initial = report_for(2000)
        _, final = report_for(completion["step"])
        best_checkpoint = torch.load(DRIVE_RUN / "best.pt", map_location="cpu", weights_only=True)
        best_folder, best = report_for(best_checkpoint["step"])
        rows = [
            ("추가 학습 전 (2,000번)", initial["mse"]["mean"], initial["foreground_mse_heuristic"]["mean"]),
            (f"최종 모델 ({completion['step']:,}번)", final["mse"]["mean"], final["foreground_mse_heuristic"]["mean"]),
            (f"보존된 최소 오차 모델 ({best_checkpoint['step']:,}번)", best["mse"]["mean"], best["foreground_mse_heuristic"]["mean"]),
            ("copy-last 기준선", final["copy_last_mse"]["mean"], final["copy_last_foreground_mse_heuristic"]["mean"]),
        ]
        table_rows = "".join(f"<tr><td>{name}</td><td>{rgb:.8f}</td><td>{fg:.6f}</td></tr>" for name, rgb, fg in rows)
        display(HTML("<table><thead><tr><th>같은 검증 영상 32개 · 미래 32프레임</th>"
                     "<th>RGB MSE ↓</th><th>물체 영역 MSE ↓</th></tr></thead><tbody>"
                     + table_rows + "</tbody></table>"))
        print(f"추가 학습 전 대비 최종 RGB 오차 감소: {(1 - final['mse']['mean'] / initial['mse']['mean']) * 100:.2f}%")
        print(f"copy-last 대비 최종 RGB 오차 감소: {(1 - final['mse']['mean'] / final['copy_last_mse']['mean']) * 100:.2f}%")
        print(f"이번 추가 업데이트: {completion['step'] - 2000:,}; 누적 업데이트: {completion['step']:,} / {completion['requested_steps']:,}")
        print(f"학습·검증 시간: {completion['wall_seconds'] / 60:.2f}분; 종료: {completion['reason']}")
        print(f"학습 GPU 최대 할당 메모리: {completion['gpu_peak_memory_gb']:.3f} GB; 실제 행동 라벨 사용: False")
        print("시점별 최종 RGB MSE:", {key: final["mse"][key] for key in ["h1", "h8", "h16", "h32"]})
        print("저장 위치:", DRIVE_RUN)
        for video in sorted(best_folder.glob("*.mp4")):
            display(Video(str(video), embed=True, width=680))
    ''')
    for index, cell in enumerate(continuation):
        cell["id"] = f"q3-continue-{index:02d}"
    path = Path(__file__).resolve().parents[1] / "notebooks" / "continue_t4_6000.ipynb"
    notebook = {"cells": continuation, "metadata": {"accelerator": "GPU", "colab": {"name": path.name},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
