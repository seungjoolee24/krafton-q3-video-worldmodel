"""Regenerate a clean, thin Colab notebook from the GitHub training entry points."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": textwrap.dedent(source).strip().splitlines(keepends=True)})


def code(source, parameters=False):
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"} if parameters else {},
                  "execution_count": None, "outputs": [],
                  "source": (textwrap.dedent(source).strip() + "\n").splitlines(keepends=True)})


markdown("""
    # Q3 · 영상만으로 학습하는 잠재 행동 월드 모델

    과거 32장의 RGB로 다음 32장을 예측합니다. 행동 `.npz`를 읽지 않습니다.
    **잠재 코드는 실제 힘과 연결되지 않았으며, 이 단계는 영상 기반 모델의 첫 실험입니다.**
    posterior는 학습 전용, 미래 예측은 과거만 보는 prior를 사용합니다.

    런타임 → 런타임 유형 변경 → **GPU**를 선택하세요.
    셀을 순서대로 실행하면 특징 캐시를 만들고, **학습 셀에서 학습을 시작**합니다.
    GitHub에는 코드, Drive에는 원본 ZIP과 결과, `/content`에는 실행용 데이터와 캐시를 둡니다.
""")
markdown("""
    ## 1. 실행 설정
    첫 실행은 `pilot`: 학습 영상 128개 + 검증 32개, 2,000 업데이트입니다.
    `full`은 학습 1,800개 + 검증 200개, 20,000 업데이트입니다.
    동일 `RUN_NAME`의 `latest.pt`가 있으면 자동 재개합니다. 새 실험은 이름을 바꾸세요.
    `MAX_STEPS`는 누적 종료 지점이며 `0`이면 설정 파일의 예산을 사용합니다.
""")
code('''
    REPO_URL = "https://github.com/seungjoolee24/krafton-q3-video-worldmodel.git" # @param {type:"string"}
    GIT_REF = "main" # @param {type:"string"}
    PROFILE = "pilot" # @param ["pilot", "full"]
    RUN_NAME = "video-only-pilot-v1" # @param {type:"string"}
    DRIVE_ROOT = "/content/drive/MyDrive/krafton-q3-video-worldmodel" # @param {type:"string"}
    DATA_ZIP = "/content/drive/MyDrive/krafton-q3-video-worldmodel/track3-kit.zip" # @param {type:"string"}
    DATA_SHA256 = "dbd9da58b386782533913da2c44b088a6c25c6119ba6efbe680dae56bf4422ee" # @param {type:"string"}
    RESUME_CHECKPOINT = "" # @param {type:"string"}
    MAX_STEPS = 0 # @param {type:"integer"}
    BATCH_SIZE = 4 # @param {type:"integer"}
''', parameters=True)
markdown("""
    ## 2. 코드와 GPU 환경
    현재 Git 커밋을 출력합니다. 재현할 때 `GIT_REF`에 이 SHA를 사용하세요.
    Colab에 설치된 CUDA Torch를 유지하고 나머지 의존성만 설치합니다.
""")
code('''
    import os, sys, subprocess, json, shutil
    from pathlib import Path
    import torch
    assert torch.cuda.is_available(), "GPU 런타임을 선택하세요."
    version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    assert version >= (2, 3), f"Torch >= 2.3 required, found {torch.__version__}"
    assert PROFILE in {"pilot", "full"} and BATCH_SIZE >= 1 and MAX_STEPS >= 0
    assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME, "RUN_NAME은 폴더 이름 하나로 지정하세요."
    CODE = Path("/content/q3-video-code")
    if not (CODE / ".git").is_dir():
        subprocess.run(["git", "clone", REPO_URL, str(CODE)], check=True)
    else:
        remote = subprocess.check_output(["git", "-C", str(CODE), "remote", "get-url", "origin"], text=True).strip()
        assert remote == REPO_URL, "Existing checkout has a different repository."
        dirty = subprocess.check_output(["git", "-C", str(CODE), "status", "--porcelain"], text=True)
        assert not dirty.strip(), "Source changes exist; preserve them before changing revision."
    subprocess.run(["git", "-C", str(CODE), "fetch", "origin", GIT_REF], check=True)
    subprocess.run(["git", "-C", str(CODE), "checkout", "--detach", "FETCH_HEAD"], check=True)
    COMMIT = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(CODE / "requirements-colab.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(CODE), "--no-deps"], check=True)
    os.chdir(CODE)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print({"commit": COMMIT, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    subprocess.run([sys.executable, "scripts/check_project.py"], check=True)
''')
markdown("""
    ## 3. Drive 연결과 원본 데이터
    Google Drive 접근은 본인 계정으로 승인합니다.
    데이터 ZIP을 런타임 로컬 디스크로 복사한 뒤 압축을 풉니다.
    학습 중 Drive에서 수천 개의 파일을 반복해서 읽지 않습니다.
""")
code('''
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path(DRIVE_ROOT)
    DRIVE.mkdir(parents=True, exist_ok=True)
    source_zip = Path(DATA_ZIP)
    LOCAL_DATA = Path("/content/q3-data")
    LOCAL_DATA.mkdir(exist_ok=True)
    local_zip = LOCAL_DATA / "track3-kit.zip"
    from importlib.util import spec_from_file_location, module_from_spec
    specification = spec_from_file_location("unpack_kit", CODE / "scripts/unpack_kit.py")
    unpack_module = module_from_spec(specification)
    specification.loader.exec_module(unpack_module)
    unpack_module.stage_archive(source_zip, local_zip, DATA_SHA256)
    KIT = unpack_module.unpack(local_zip, LOCAL_DATA / "unpacked")
    CACHE = Path(f"/content/q3-cache-{PROFILE}")
    LOCAL_RUN = Path(f"/content/q3-runs/{RUN_NAME}")
    DRIVE_RUN = DRIVE / "runs" / RUN_NAME
    config = json.loads((CODE / "configs" / f"{PROFILE}.json").read_text())
    config["batch_size"] = BATCH_SIZE
    if MAX_STEPS:
        config["total_steps"] = MAX_STEPS
    LOCAL_RUN.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = LOCAL_RUN / "config.json"
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print({"kit": str(KIT), "cache": str(CACHE), "drive_results": str(DRIVE_RUN), "config": config})
''')
markdown("""
    ## 4. 짧은 동작 검증
    작은 합성 입력의 gradient·배치 독립성·미래 정보 차단·재개를 검사합니다.
    GPU smoke는 두 번의 작은 합성 업데이트만 수행하며 학습 가중치를 저장하지 않습니다.
""")
code('''
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
    subprocess.run([sys.executable, "-m", "video_wam.cli", "smoke", "--kit", str(KIT), "--device", "cuda"], check=True)
''')
markdown("""
    ## 5. 영상 특징 캐시
    행동 sidecar는 읽지 않습니다. 원본 RGB를 제공된 고정 인코더로 변환합니다.
    Pilot는 약 3GB, 전체는 약 35GB의 캐시 공간이 필요합니다.
    중단된 같은 런타임에서 다시 실행하면 검증된 에피소드 캐시를 재사용합니다.
    런타임이 삭제되면 원본 ZIP에서 다시 생성합니다.
""")
code('''
    subprocess.run([sys.executable, "-m", "video_wam.cli", "prepare",
                    "--kit", str(KIT), "--cache", str(CACHE), "--config", str(CONFIG_PATH),
                    "--encode-batch", "64", "--device", "cuda"], check=True)
    index = json.loads((CACHE / "index.json").read_text())
    print({"train_episodes": len(index["train_ids"]), "dev_episodes": len(index["dev_ids"]),
           "actions_used": index["actions_used"], "cache_fingerprint": index["fingerprint"]})
''')
markdown("""
    ## 6. GPU 학습 시작 또는 재개
    이 셀을 실행하면 설정한 예산까지 학습합니다. 실제 행동 라벨은 사용하지 않습니다.
    latest/best/final 체크포인트와 검증 영상은 Drive에 저장합니다.
    동일 실행의 체크포인트로 재개할 때는 모델·배치·분할·학습 목적을 유지하세요.
    첫 로그의 처리량을 기준으로 이후 학습 시간을 판단하세요.
""")
code('''
    resume = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else DRIVE_RUN / "latest.pt"
    command = [sys.executable, "-m", "video_wam.cli", "train", "--kit", str(KIT),
               "--cache", str(CACHE), "--config", str(CONFIG_PATH), "--out", str(LOCAL_RUN),
               "--persist", str(DRIVE_RUN), "--device", "cuda"]
    if RESUME_CHECKPOINT:
        assert resume.is_file(), f"Resume checkpoint not found: {resume}"
    if resume.is_file():
        command.extend(["--resume", str(resume)])
        print("Resume:", resume)
    if MAX_STEPS:
        command.extend(["--max-steps", str(MAX_STEPS)])
    subprocess.run(command, check=True)
''')
markdown("""
    ## 7. 32프레임 검증과 비교 영상
    실제 미래를 입력하지 않는 prior의 예측만 평가합니다.
    RGB MSE·PSNR·물체 영역 보조 오차와 copy-last 기준선을 비교합니다.
    이 결과는 실제 힘을 주었을 때의 반응 정확도나 공식 점수가 아닙니다.
""")
code('''
    from IPython.display import display, Video
    CHECKPOINT = DRIVE_RUN / "best.pt"
    assert CHECKPOINT.is_file(), "학습 또는 체크포인트 재개를 먼저 실행하세요."
    REPORT = LOCAL_RUN / "final-report"
    subprocess.run([sys.executable, "-m", "video_wam.cli", "evaluate", "--kit", str(KIT),
                    "--cache", str(CACHE), "--checkpoint", str(CHECKPOINT), "--out", str(REPORT),
                    "--horizon", "32", "--previews", "3", "--device", "cuda"], check=True)
    metrics = json.loads((REPORT / "metrics.json").read_text())
    display({key: value for key, value in metrics.items() if key != "per_episode"})
    destination = DRIVE_RUN / "final-report"
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in REPORT.iterdir():
        shutil.copy2(artifact, destination / artifact.name)
    for video in sorted(REPORT.glob("*.mp4")):
        display(Video(str(video), embed=True, width=768))
    print("Saved:", destination)
''')


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "train_video_only_colab.ipynb"
    path.parent.mkdir(exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = f"q3-video-{index:02d}"
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": path.name},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}},
                "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
