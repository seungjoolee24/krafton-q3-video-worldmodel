"""Generate the larger T4 run, initialized from the first validation model."""
from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

from make_notebook import cells


def main():
    expanded = copy.deepcopy(cells)

    def replace(index, source):
        expanded[index]["source"] = (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)

    replace(0, """
        # Q3 · T4 데이터·학습 확장

        **학습 영상 128개 · 검증 영상 32개 · 추가 2,000 업데이트**를 실행합니다.
        첫 검증(16/4 영상, 300 업데이트)의 best.pt에서 모델 가중치만 가져옵니다.
        데이터 선택이 달라지므로 optimizer·난수·step은 새로 시작합니다.
        같은 검증 영상 32개에서 시작 가중치, 학습 후 모델, copy-last를 비교합니다.
        실제 행동 라벨은 사용하지 않습니다. 예측은 과거 32장 → 미래 32장입니다.
    """)
    replace(1, """
        ## 1. 확장 실험 설정
        첫 검증 결과는 별도 폴더에 보존하고 새 실행 폴더에 저장합니다.
        공식 에피소드별 train/dev 구분을 유지합니다. 선택 영상 수를 늘리면 부분집합이 달라지므로
        이전 검증 4개의 평균과 새 검증 32개의 평균을 직접 비교하지 않습니다.
        같은 이름의 latest.pt가 있으면 이 확장 실험을 재개합니다.
    """)
    replace(2, '''
        import time
        REPO_URL = "https://github.com/seungjoolee24/krafton-q3-video-worldmodel.git"
        GIT_REF = "main"
        PROFILE = "pilot"
        RUN_NAME = "video-only-t4-expanded-v1"
        DRIVE_ROOT = "/content/drive/MyDrive/krafton-q3-video-worldmodel"
        DATA_ZIP = DRIVE_ROOT + "/track3-kit.zip"
        DATA_SHA256 = "dbd9da58b386782533913da2c44b088a6c25c6119ba6efbe680dae56bf4422ee"
        INIT_CHECKPOINT = DRIVE_ROOT + "/runs/video-only-t4-first-v1/best.pt"
        RESUME_CHECKPOINT = ""
        MAX_STEPS = 2000
        BATCH_SIZE = 4
        EXPANDED_STARTED = time.time()
        print({"train_videos": 128, "dev_videos": 32, "new_updates": MAX_STEPS,
               "initialize_weights_from": INIT_CHECKPOINT, "actions_used": False})
    ''')
    expanded[2]["metadata"] = {}
    data = "".join(expanded[6]["source"])
    data = data.replace('config["batch_size"] = BATCH_SIZE',
                        'config["batch_size"] = BATCH_SIZE\nconfig["evaluate_at_start"] = True')
    replace(6, data)
    replace(11, """
        ## 6. 추가 2,000번 학습
        이어받은 모델을 먼저 새로운 검증 영상 32개에서 평가합니다.
        이후 1 → 4 → 8 → 16 → 32 프레임의 예측 길이로 추가 학습합니다.
        새 데이터에 맞춰 optimizer와 KL 워밍업을 초기화합니다.
        이 확장 실험의 latest.pt에서 재개할 때는 optimizer·step·난수도 복원합니다.
        100번마다 저장하고 200번마다 32개 영상을 검증하여 Drive에 보관합니다.
    """)
    training = "".join(expanded[12]["source"])
    training = training.replace('    print("Resume:", resume)', '''    print("Resume:", resume)
else:
    initial_checkpoint = Path(INIT_CHECKPOINT)
    assert initial_checkpoint.is_file(), f"Initial weights not found: {initial_checkpoint}"
    command.extend(["--init-from", str(initial_checkpoint)])
    print("Initialize model weights only:", initial_checkpoint)''')
    replace(12, training)
    replace(13, """
        ## 7. 같은 32개 영상에서 학습 전후 비교
        최종 모델과 검증 오차가 가장 낮았던 best.pt를 구분하여 보여줍니다.
        RGB MSE는 전체 픽셀의 오차, 물체 영역 MSE는 색을 사용하는 보조 지표입니다.
        비교 영상은 검증 오차가 가장 낮았던 모델의 예측입니다. 공식 점수나 힘 복원 정확도는 아닙니다.
    """)
    replace(14, '''
        from IPython.display import display, HTML, Video
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

        _, initial = report_for(0)
        final_folder, final = report_for(completion["step"])
        best_checkpoint = torch.load(DRIVE_RUN / "best.pt", map_location="cpu", weights_only=True)
        best_folder, best = report_for(best_checkpoint["step"])
        rows = [
            ("학습 전 (첫 검증 가중치)", initial["mse"]["mean"], initial["foreground_mse_heuristic"]["mean"]),
            (f"최종 모델 (추가 {completion['step']:,}회)", final["mse"]["mean"], final["foreground_mse_heuristic"]["mean"]),
            (f"최소 검증 오차 모델 ({best_checkpoint['step']:,}회)", best["mse"]["mean"], best["foreground_mse_heuristic"]["mean"]),
            ("copy-last 기준선", final["copy_last_mse"]["mean"], final["copy_last_foreground_mse_heuristic"]["mean"]),
        ]
        table_rows = "".join(f"<tr><td>{name}</td><td>{rgb:.8f}</td><td>{fg:.6f}</td></tr>"
                             for name, rgb, fg in rows)
        display(HTML("<table><thead><tr><th>검증 영상 32개 · 미래 32프레임</th>"
                     "<th>RGB MSE ↓</th><th>물체 영역 MSE ↓</th></tr></thead><tbody>"
                     + table_rows + "</tbody></table>"))
        print(f"학습 전 대비 최종 RGB 오차 감소: {(1 - final['mse']['mean'] / initial['mse']['mean']) * 100:.2f}%")
        print(f"copy-last 대비 최종 RGB 오차 감소: {(1 - final['mse']['mean'] / final['copy_last_mse']['mean']) * 100:.2f}%")
        print(f"완료 업데이트: {completion['step']:,} / {completion['requested_steps']:,}; 종료: {completion['reason']}")
        print(f"학습·검증 시간: {completion['wall_seconds'] / 60:.2f}분; 전체 실행: {(time.time() - EXPANDED_STARTED) / 60:.2f}분")
        print(f"학습 GPU 최대 할당 메모리: {completion['gpu_peak_memory_gb']:.3f} GB; 실제 행동 라벨 사용: False")
        print("시점별 최종 RGB MSE:", {key: final["mse"][key] for key in ["h1", "h8", "h16", "h32"]})
        print("저장 위치:", DRIVE_RUN)
        for video in sorted(best_folder.glob("*.mp4")):
            display(Video(str(video), embed=True, width=680))
    ''')
    for index, cell in enumerate(expanded):
        cell["id"] = f"q3-expanded-{index:02d}"
    path = Path(__file__).resolve().parents[1] / "notebooks" / "train_t4_expanded.ipynb"
    notebook = {"cells": expanded, "metadata": {"accelerator": "GPU", "colab": {"name": path.name},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
