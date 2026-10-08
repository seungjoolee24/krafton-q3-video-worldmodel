"""Generate the labelled-only, action-conditioned T4 launcher notebook."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


cells: list[dict] = []


def markdown(source: str):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})


def code(source: str):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [],
                  "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})


markdown("""
    # Q3 · 행동 + 프레임 변화량 · T4 기본 모델

    행동 라벨이 있는 **전체 200개 영상**을 기존 에피소드 분할에 맞춰
    **185개 학습 / 15개 검증**에 사용합니다. 비라벨 영상과 이전 영상 전용 모델의 가중치는 사용하지 않습니다.
    제공된 고정 시각 인코더·디코더는 사용하며, 새 행동 조건부 전이 모델을 처음부터 학습합니다.

    과거 RGB 32장과 과거 행동 31개로 상태를 추정하고, **주어진 미래 행동 32개**로 미래 RGB 32장을 예측합니다.
    특징 차이 `f[t]-f[t-1]`에는 **이전 행동 `a[t-1]`**을 연결합니다.
    1·4·8프레임 특징 차이를 사용하며, 그 사이의 모든 프레임과 행동을 차례로 입력해 ConvGRU가 관측 이력을 기억합니다.
    미래 정답은 손실 계산에만 사용하고, 자율 예측 상태에 입력하지 않습니다.

    **런타임 → 런타임 유형 변경 → T4 GPU**를 선택하고 셀을 순서대로 실행하세요.
    코드는 GitHub에서 가져오고, 학습은 Colab GPU, 원본과 결과 보관은 Drive에서 수행합니다.
    이 노트북은 학습 셀을 실행하기 전까지 본 학습을 시작하지 않습니다.
""")

markdown("""
    ## 1. 실행 설정

    이 실험은 새 모델·새 결과 폴더를 사용합니다. 동일 `RUN_NAME`에 저장된 행동 모델의
    `latest.pt`가 있으면 그 실행만 재개합니다. 이전 영상 전용 체크포인트와 섞지 마세요.
    전체 200개 중 검증 15개는 gradient 계산에 사용하지 않습니다.
""")

code('''
    REPO_URL = "https://github.com/seungjoolee24/krafton-q3-video-worldmodel.git"
    GIT_REF = "main"
    RUN_NAME = "action-difference-t4-v1"
    DRIVE_ROOT = "/content/drive/MyDrive/krafton-q3-video-worldmodel"
    DATA_ZIP = DRIVE_ROOT + "/data/track3-labelled-200.zip"
    DATA_SHA256 = ""  # Empty: read the archive's .json sidecar checksum.
    RESUME_CHECKPOINT = ""  # Optional: this action-model run's latest.pt only.
    MAX_STEPS = 4000
    BATCH_SIZE = 4
    print({"train_videos": 185, "dev_videos": 15, "actions_used": True,
           "context_frames": 32, "future_actions": 32, "max_steps": MAX_STEPS,
           "difference_lags": [1, 4, 8]})
''')

markdown("""
    ## 2. 코드와 T4 환경

    실행할 Git 커밋을 한 번 선택하고 기록합니다. 재현할 때 `GIT_REF`에 출력된 SHA를 넣으세요.
    Colab의 CUDA Torch를 유지하고 나머지 의존성만 설치합니다.
    실행 도중 GitHub 코드를 수정하면 이 런타임을 다시 설정한 뒤 새 실험 이름으로 실행하세요.
""")

code('''
    import os, sys, subprocess, json, importlib
    from pathlib import Path
    import torch
    assert torch.cuda.is_available(), "런타임 설정에서 T4 GPU를 선택하세요."
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    assert version >= (2, 3), f"Torch >= 2.3 required, found {torch.__version__}"
    assert BATCH_SIZE >= 1 and MAX_STEPS >= 1
    assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME, "RUN_NAME은 폴더 이름 하나로 지정하세요."
    CODE = Path("/content/q3-action-code")
    if not (CODE / ".git").is_dir():
        subprocess.run(["git", "clone", REPO_URL, str(CODE)], check=True)
    else:
        remote = subprocess.check_output(["git", "-C", str(CODE), "remote", "get-url", "origin"], text=True).strip()
        assert remote == REPO_URL, "Existing checkout belongs to a different repository."
        dirty = subprocess.check_output(["git", "-C", str(CODE), "status", "--porcelain"], text=True)
        assert not dirty.strip(), "Preserve local source changes before selecting a revision."
    subprocess.run(["git", "-C", str(CODE), "fetch", "origin", GIT_REF], check=True)
    subprocess.run(["git", "-C", str(CODE), "checkout", "--detach", "FETCH_HEAD"], check=True)
    COMMIT = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(CODE / "requirements-colab.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(CODE), "--no-deps"], check=True)
    # The current notebook kernel does not re-read editable-install .pth files.
    SOURCE_PATH = str(CODE / "src")
    if SOURCE_PATH not in sys.path:
        sys.path.insert(0, SOURCE_PATH)
    importlib.invalidate_caches()
    import action_wam
    assert Path(action_wam.__file__).resolve().is_relative_to(CODE.resolve())
    os.chdir(CODE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print({"commit": COMMIT, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
''')

markdown("""
    ## 3. Drive 연결과 라벨 데이터

    Drive의 `data/track3-labelled-200.zip`에는 행동이 있는 200개 영상·행동 파일과 제공된 시각 stem이 들어 있습니다.
    ZIP 한 개 또는 업로드 크기에 맞춘 분할 파일 두 개에서 **약 175MB의 라벨 데이터 ZIP만** 런타임 디스크에 구성하고 체크섬을 확인합니다.
    파일이 없으면 라벨 ZIP 또는 분할 파일·manifest·체크섬 업로드를 먼저 완료하세요.
    원본 전체 2,000개 영상을 내려받는 방식은 사용하지 않습니다.
""")

code('''
    from google.colab import drive
    from importlib.util import spec_from_file_location, module_from_spec
    drive.mount("/content/drive", timeout_ms=60000)
    DRIVE = Path(DRIVE_ROOT)
    source_zip = Path(DATA_ZIP)
    parts_manifest = source_zip.parent / "track3-kit.parts.json"
    if not source_zip.is_file() and not parts_manifest.is_file():
        raise FileNotFoundError(f"라벨 영상 200개 ZIP 또는 분할 파일과 manifest를 Drive에 업로드하세요: {source_zip} / {parts_manifest}")
    checksum = DATA_SHA256.strip()
    if not checksum:
        sidecar = source_zip.with_suffix(source_zip.suffix + ".json")
        if not sidecar.is_file():
            sidecar = source_zip.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"데이터 SHA-256을 DATA_SHA256에 넣거나 checksum sidecar를 업로드하세요: {sidecar}")
        archive_info = json.loads(sidecar.read_text())
        checksum = archive_info.get("sha256") or archive_info.get("archive_sha256")
    assert isinstance(checksum, str) and len(checksum) == 64 and all(c in "0123456789abcdefABCDEF" for c in checksum), "Valid SHA-256 required."
    LOCAL_DATA = Path("/content/q3-action-data")
    LOCAL_DATA.mkdir(exist_ok=True)
    specification = spec_from_file_location("unpack_kit", CODE / "scripts/unpack_kit.py")
    unpack_module = module_from_spec(specification)
    specification.loader.exec_module(unpack_module)
    local_zip = LOCAL_DATA / "track3-labelled-200.zip"
    unpack_module.stage_archive(source_zip, local_zip, checksum.lower())
    KIT = unpack_module.unpack(local_zip, LOCAL_DATA / "unpacked")
    CACHE = Path("/content/q3-action-cache")
    LOCAL_RUN = Path("/content/q3-action-runs") / RUN_NAME
    DRIVE_RUN = DRIVE / "runs" / RUN_NAME
    config = json.loads((CODE / "configs/action_t4.json").read_text())
    config["batch_size"] = BATCH_SIZE
    config["total_steps"] = MAX_STEPS
    assert config["labelled_only"] and config["train_limit"] is None and config["dev_limit"] is None
    LOCAL_RUN.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = LOCAL_RUN / "config.json"
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print({"kit": str(KIT), "cache": str(CACHE), "drive_results": str(DRIVE_RUN),
           "data_sha256": checksum, "config": config})
''')

markdown("""
    ## 4. GPU 동작 검증

    작은 합성 입력으로 프레임·행동 정렬, 미래 정보 차단, gradient, 배치 독립성, 재개 동작을 확인합니다.
    이 검사는 본 학습의 체크포인트를 생성하지 않습니다.
""")

code('''
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                    "-p", "test_action*.py", "-v"], check=True)
    subprocess.run([sys.executable, "-m", "action_wam.cli", "smoke",
                    "--kit", str(KIT), "--device", "cuda"], check=True)
''')

markdown("""
    ## 5. 라벨 영상 전체의 특징·RGB 캐시

    제공된 인코더로 모든 라벨 영상의 특징을 한 번 계산합니다. 행동값과 RGB도 함께 저장합니다.
    특징은 FP16, RGB는 uint8로 보관해 T4 학습 중 반복 디코딩을 피합니다.
    인코더·디코더 가중치는 고정하지만 RGB·윤곽 손실의 gradient는 예측 특징으로 흐릅니다.
    동일한 런타임에서 재실행하면 검증된 캐시를 재사용합니다.
""")

code('''
    subprocess.run([sys.executable, "-m", "action_wam.cli", "prepare",
                    "--kit", str(KIT), "--cache", str(CACHE), "--config", str(CONFIG_PATH),
                    "--encode-batch", "64", "--device", "cuda"], check=True)
    index = json.loads((CACHE / "index.json").read_text())
    assert index["actions_used"] is True
    assert len(index["train_ids"]) == 185 and len(index["dev_ids"]) == 15, "Expected all 200 labelled episodes and original split."
    assert set(index["train_ids"]).isdisjoint(index["dev_ids"]), "Train/dev overlap."
    print({"train_episodes": len(index["train_ids"]), "dev_episodes": len(index["dev_ids"]),
           "all_labelled_episodes": len(index["train_ids"]) + len(index["dev_ids"]),
           "actions_used": index["actions_used"], "cache_fingerprint": index["fingerprint"],
           "model": config["model"], "horizon_schedule": config["horizon_schedule"]})
''')

markdown("""
    ## 6. 행동 조건부 학습 시작 또는 재개

    이 셀을 실행하면 본 학습을 시작합니다. Batch 4, FP16, 총 4,000 업데이트가 기본입니다.
    예측 길이를 1 → 4 → 8 → 16 → 32프레임으로 늘립니다. 특징 손실에 RGB·윤곽 손실을 추가합니다.
    250번마다 저장하고, 500번마다 같은 검증 영상 15개를 미래 32프레임까지 평가합니다.
    학습 영상 185개 모두가 표본 추출 대상이며 실제 관측한 에피소드 수도 완료 파일에 기록합니다.

    같은 실행의 `latest.pt`가 있으면 모델·optimizer·난수·step을 복원합니다.
    체크포인트는 행동 모델·같은 데이터 분할·같은 학습 설정인지 검사합니다.
    중단 버튼을 누르면 마지막 완료된 업데이트를 저장합니다.
""")

code('''
    import signal
    resume = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else DRIVE_RUN / "latest.pt"
    command = [sys.executable, "-u", "-m", "action_wam.cli", "train",
               "--kit", str(KIT), "--cache", str(CACHE), "--config", str(CONFIG_PATH),
               "--out", str(LOCAL_RUN), "--persist", str(DRIVE_RUN), "--device", "cuda"]
    if RESUME_CHECKPOINT and not resume.is_file():
        raise FileNotFoundError(f"Explicit resume checkpoint is missing: {resume}")
    if resume.is_file():
        command.extend(["--resume", str(resume)])
        print("Resume this action-model run:", resume, flush=True)
    else:
        print("New action-conditioned model; no video-only checkpoint used.", flush=True)
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

markdown("""
    ## 7. 저장된 결과와 비교 영상

    별도의 추가 학습·추가 검증 없이 학습 중 저장한 결과를 읽습니다.
    최종 모델과 RGB MSE가 가장 낮은 모델을 구분합니다. 물체 영역 지표는 색 기반 보조 오차입니다.
    행동을 바꿨을 때의 출력 변화와 틀린 행동의 예측 오차는 행동 활용을 확인하는 보조 진단이며,
    실제 다른 행동을 시행한 정답 영상에 대한 검증은 아닙니다. 공식 점수와 구분해 해석하세요.
    런타임 초기화 후에도 이 결과 셀만 실행해 Drive 결과를 열 수 있습니다.
""")

code('''
    import json
    from pathlib import Path
    from google.colab import drive
    from IPython.display import display, HTML, Video
    drive.mount("/content/drive", timeout_ms=60000)
    if "DRIVE_RUN" not in globals():
        DRIVE_RUN = Path("/content/drive/MyDrive/krafton-q3-video-worldmodel/runs/action-difference-t4-v1")
    if "LOCAL_RUN" not in globals():
        LOCAL_RUN = Path("/content/q3-action-runs/action-difference-t4-v1")
    completion_path = LOCAL_RUN / "completion.json"
    if not completion_path.is_file():
        completion_path = DRIVE_RUN / "completion.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"완료 파일이 아직 없습니다. 학습 셀 출력을 확인하세요: {completion_path}")
    completion = json.loads(completion_path.read_text())

    def report_for(step):
        relative = Path("validation") / f"step_{step:06d}"
        folder = LOCAL_RUN / relative
        if not (folder / "metrics.json").is_file():
            folder = DRIVE_RUN / relative
        return folder, json.loads((folder / "metrics.json").read_text())

    final_folder, final = report_for(completion["step"])
    report_candidates = {}
    for base in [DRIVE_RUN, LOCAL_RUN]:
        for path in sorted((base / "validation").glob("step_*/metrics.json")):
            report = json.loads(path.read_text())
            report_candidates[int(path.parent.name.split("_")[-1])] = (path.parent, report)
    best_step, (best_folder, best) = min(report_candidates.items(), key=lambda item: item[1][1]["mse"]["mean"])
    rows = [
        (f"최종 모델 ({completion['step']:,}번)", final["mse"]["mean"], final["foreground_mse_heuristic"]["mean"]),
        (f"최소 RGB 오차 모델 ({best_step:,}번)", best["mse"]["mean"], best["foreground_mse_heuristic"]["mean"]),
        ("Copy last", final["copy_last_mse"]["mean"], final["copy_last_foreground_mse_heuristic"]["mean"]),
    ]
    table_rows = "".join(f"<tr><td>{name}</td><td>{rgb:.8f}</td><td>{fg:.6f}</td></tr>" for name, rgb, fg in rows)
    display(HTML("<table><thead><tr><th>검증 영상 15개 · 미래 32프레임</th>"
                 "<th>RGB MSE ↓</th><th>물체 영역 MSE ↓</th></tr></thead><tbody>"
                 + table_rows + "</tbody></table>"))
    print(f"완료 업데이트: {completion['step']:,} / {completion['requested_steps']:,}; 종료: {completion['reason']}")
    print(f"학습·검증 시간: {completion['wall_seconds'] / 60:.2f}분")
    print(f"Copy last 대비 최종 RGB 오차 감소: {(1 - final['mse']['mean'] / final['copy_last_mse']['mean']) * 100:.2f}%")
    print("시점별 RGB MSE:", {key: final["mse"][key] for key in ["h1", "h8", "h16", "h32"]})
    for key in ["actual_updates", "train_episodes_seen", "gpu_peak_memory_gb"]:
        if key in completion:
            print(key, completion[key])
    for key in ["wrong_action_mse", "action_sensitivity"]:
        if key in final:
            print(key, final[key])
    print("실제 행동 라벨 사용: True; 저장 위치:", DRIVE_RUN)
    for video in sorted(best_folder.glob("*.mp4")):
        display(Video(str(video), embed=True, width=720))
''')


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/train_action_difference_t4.ipynb"
    path.parent.mkdir(exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = f"q3-action-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": path.name,
                "gpuType": "T4"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
