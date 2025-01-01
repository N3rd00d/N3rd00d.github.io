---
title: "Flux Lora 훈련 후기와 RunPod 사용법"
description: "Flux Lora 훈련법과 RunPod 사용법을 간단히 알아보자."
date: 2024-12-31T19:38:40.841Z
tags: ["Flux","Lora","RunPod","Stable Diffusion","traning"]
---
# 들어가며
저자의 현재 데스크탑의 사양은 다음과 같다.
```
|-----|-------------|
| CPU | Ryzen 2700X |
|-----|-------------|
| GPU | RTX 4070Ti  |
|-----|-------------|
| RAM | 32GB        |
|-----|-------------|
```

훈련은 3000스텝으로 설정, 그 외엔 기본 오프셋을 사용.
```
|------------|------------|------------|------|
| Base Model | Trainer    | VRAM       | Time |
|------------|------------|------------|------|
| SD1.5      | Kohya      | 8GB        | 1.5h |
|------------|------------|------------|------|
| SDXL       | OneTrainer | 12GB       | 4.5h |
|------------|------------|------------|------|
| Flux1.d    | FluxGym    | 12/16/20GB | 11h  |
|------------|------------|------------|------|
```

Lora는 도트64스타일의 2등신의 치비 캐릭터 생성을 목표로 하였다. 다음은 각 베이스모델별 간단한 훈련/생성 후기이다.

 - SD1.5 - 샘플 품질 낮음.
![](/images/89c225da-0de9-40d1-a038-b39ae530bcef-image.png)
 - SDXL - 데스크탑 풀로드, Kohya에서 OOM발생으로 OneTrainer로 교체하여 훈련진행, 샘플링 결과물은 그럭저럭이나 도트 스타일이 균일하지 않음.(본디 SDXL이 그런 경향을 보임)   
![](/images/f52a0cc9-c369-491d-805c-04f1ba4ecb91-image.png)
![](/images/f8b1e910-33b9-45f3-8e12-fbb1b15d4579-image.png)
 - Flux1.d - 최소 VRAM 24GB을 요구하나 FluxGym VRAM 12GB로 맞추고 훈련 시작, 소요시간이 11시간이 걸려 중도포기

Flux1.d 부터는 지금의 GPU로는 어렵다고 판단, Flux Lora를 만들기 위해 클라우드 GPU 대여 서비스 RunPod 사용.

# [RunPod](https://www.runpod.io/) 포드 생성 및 접속

## 1. 비용 충전
금액을 사전에 충전해야한다. $10부터 충전 가능하다.

![](/images/430e91d7-6fbc-414d-b6de-c753b548f175-image.png)

## 2. 대여할 GPU 선택
RunPod에서 제공하는 GPU는 현재 기준으로 다음과 같다. Secure/Community 두가지 컴퓨팅 서비스를 제공하고 있다. 두 서비스의 차이는 [공식](https://docs.runpod.io/references/faq)을 참고. 저자는 저렴한 Community Cloud를 주로 사용함.

![](/images/68017289-4579-4ba8-aaec-b106dce5b74c-image.png)

여러 GPU들이 있지만, 최근 생상된 GPU 또는 VRAM이 많을 수록 대여가격이 올라간다. 저자는 주로 A40과 RTX4090을 사용. RTX4090은VRAM 24GB로 OOM이 발생되긴하나, 가격 대비 성능이 우수하다. 

참고
- GPU별 Flux Lora 훈련 소요시간
```
|---------|------|------------|-------|----------------------|
| GPU     | VRAM | VRAM share | Time  | Model                |
|---------|------|------------|-------|----------------------|
| A40     | 48   | 50%        | 02:05 | pixel_sprite_lora_v1 |
|---------|------|------------|-------|----------------------|
| RTX4090 | 24   | 93%        | 01:05 | pixel_sprite_lora_v1 |
|---------|------|------------|-------|----------------------|
```

## 3. 도커와 인스턴스 선택하기

### 3.1 도커 이미지
RunPod는 생성형 인공지능 훈련 및 생성을 위한 다양한 도커 이미지를 제공한다. Change Template 버튼을 눌러 준비된 이미지를 선택할 수 있다. Flux 훈련엔 "AI Toolkit - ostris - ai-toolkit - officlal", Flux ComfyUI는 "ComfyUI with Flux.1 dev one-click"(ComfyUI 수동 업데이트 필요) 이미지를 사용.

![](/images/5096a736-224f-4968-a4a3-b9857ed27064-image.png)

### 3.2 포드 템플릿 수정
Edit Template 버튼을 누르면 실행할 이미지의 디스크사이즈, 포트 등을 설정할 수 있다. Container Disk는 임시저장공간이며 포드가 중지되면 Volume Disk를 제외한 데이터는 소실된다. Volume Disk에는 훈련 또는 실행에 필요한 소스코드들과 모델이 배치된다. 저자는 포드나 환경변수를 수정할 일은 없고, Container/Volume Disk를 넉넉하게 100~120GB 정도 설정해서 사용한다.

![](/images/7f4fe850-1d23-4698-b824-b1929e7a12eb-image.png)

### 3.3 인스턴스 선택
마지막으로 On-Demand/Spot 인스턴스중 하나를 선택할 수 있다. Spot 인스턴스는 비용이 저렴하나 요청이 몰려 클라우드 자원이 부족해지면 언제든 회수될 수 있는 인스턴스이다. On-Demand 사용을 권장한다.

## 4. Pod 배포 및 접속

최하단 배포버튼을 누르면 Pods 화면으로 넘어가며 이미지 설치가 자동으로 진행된다. 설치가 완료되어 연결이 가능한 시점부터 비용이 청구된다.

![](/images/681e634f-090b-4f0a-b34e-30f8542c596e-image.png)

설치가 완료되면 Connect 버튼이 생기며, 원격서버로 접속할 수 있는 방법이 제공된다.

![](/images/5f83a98a-9630-432d-b98d-38c89705a6f1-image.png)

웹 GUI를 사용하고자 한다면, 원하는 포트를 선택해서 웹링크를 열 수 있다. 8188-ComfyUI, 8888-주피터노트북 디폴트 포트 그대로 사용한다.

### Basic SSH Terminal
ssh 터미널을 사용할 땐 2가지 옵션이 제공된다. Basic SSH Termianl은 사용이 간편하나, 사전에 ssh 퍼블릭키 등록이 필요하다. 등록은 [설정](https://www.runpod.io/console/user/settings)탭에서 쉽게 등록할 수 있다.

![](/images/5660a447-5a26-4798-b1b1-4c246189d10a-image.png)

### SSH over exposed TCP
SCP/SFTP 사용하고 싶다면 SSH over exposed TCP를 사용해야한다. 매번 인스턴스를 실행할 때마다 root와 sshd 설정을 해야해서 번거로우니 파일전송이 필요하다면 [runpodctl](https://github.com/runpod/runpodctl) 사용을 추천한다. 추가 설정은 [블로그](https://blog.runpod.io/accessing-a-runpod-pod-via-ssh-username-password/) 을 참고한다. 

### 원격 서버 접속
ssh 터미널을 통해 접속하였다면, 다음과 같이 출력될 것이다.
```shell
ssh r6vkjsgoum484y-some_address@ssh.runpod.io -i ~/.ssh/id_ed25519


-- RUNPOD.IO --
Enjoy your Pod #r6vkjsgoum484y ^_^



 ______                 ______            _
(_____ \               (_____ \          | |
 _____) ) _   _  ____   _____) )___    __| |
|  __  / | | | ||  _ \ |  ____// _ \  / _  |
| |  \ \ | |_| || | | || |    | |_| |( (_| |
|_|   |_||____/ |_| |_||_|     \___/  \____|

For detailed documentation and guides, please visit:
https://docs.runpod.io/ and https://blog.runpod.io/


root@some_id_blahblah:/workspace# cd /workspace/ai-toolkit
root@some_id_blahblah:/workspace/ai-toolkit# ls
FAQ.md     assets                      docker               flux_train_ui.py  notebooks     requirements.txt  scripts
LICENSE    build_and_push_docker.yaml  extensions           info.py           output        run.py            testing
README.md  config                      extensions_built_in  jobs              repositories  run_modal.py      toolkit
root@some_id_blahblah:/workspace/ai-toolkit#
```

# Lora 훈련 세팅 및 시작
[AI-Toolkit](https://github.com/ostris/ai-toolkit)은 Flux 훈련을 위한 오픈소스 프로젝트이다. 이를 사용하기 전엔 [FluxGym](https://github.com/cocktailpeanut/fluxgym)을 사용했으나 데이터세트 업로드가 실패하는 이슈가 있어 분기 전 프로젝트인 ai-toolkit을 선택. Gradio UI도 제공하고 있으니 터미널 작업이 불편한 이들은 이를 사용할 수도 있겠다.

참고
- https://github.com/ostris/ai-toolkit?tab=readme-ov-file#training-in-modal

## 1. 허깅페이스 토큰 발급해서 적용
https://huggingface.co/settings/tokens 에 접속하여 새로운 토큰을 발급받는다. 이 토큰은 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 모델을 다운로드하는데 쓰인다. 저자는 저장소 권한에서 black-forest-labs/FLUX.1-dev 저장소만 검색 & 선택하여 Read access to contents of selected repos 항목만 활성화하여 토큰을 생성.
![](/images/822d1e05-40d1-4970-a12c-d02808688d3a-image.png)

생성된 토큰을 복사하여, 인스턴스의 `/workspace/ai-toolkit/.env` 파일을 생성하고 다음과 같이 토큰 정보를 저장한다.vim을 사용하거나 로컬에서 작성하고 [runpodctl](https://github.com/runpod/runpodctl) 로 옮기는 것도 좋을 것이다.
```
HF_TOKEN=hf_...ziXv
```

## 2. config.yaml 만들기
참고
- https://civitai.com/articles/7097/flux-complete-lora-settings-and-dataset-guide-post-mortem-of-two-weeks-of-learning
- https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_flux_24gb.yaml

훈련설정파일이 필요하다. 샘플은 `/workspace/ai-toolkit/config/examples` 디렉토리를 참고하여라. 여러 후기들을 참고해보고 테스트 해본 결과, steps 2500~3000에서 만족스럽고, 옵티마이저는 Prodigy를 사용하는 것이 괜찮았다. 참고로 디폴트 옵션인 adamw8bit와 Prodigy간 훈련시간은 대동소이하다.
```
|-----------|---------|---------|-------|----------|
| optimizer | GPU     | samples | time  | s/it     |
|-----------|---------|---------|-------|----------|
| adamw8bit | RTX4090 | 18      | 01:05 | 1.22s/it |
|-----------|---------|---------|-------|----------|
| Prodigy   | RTX4090 | 18      | 01:10 | 1.46s/it |
|-----------|---------|---------|-------|----------|
```

저자는 아래 세팅으로 훈련을 진행하였다.
```yaml
---
job: extension
config:
  name: "pixel_sprite_lora_v1"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "p1x3l"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16 
        save_every: 250 
        max_step_saves_to_keep: 1000 
        push_to_hub: false
      datasets:
        - folder_path: "/workspace/datasets"
          caption_ext: "txt"
          caption_dropout_rate: 0.05  
          shuffle_tokens: false  
          cache_latents_to_disk: true  
          resolution: [ 512, 768, 1024 ]  
      train:
        batch_size: 1
        steps: 3000  
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false  
        gradient_checkpointing: true  
        noise_scheduler: "flowmatch" 
        optimizer: "Prodigy"
        lr: 1
        noise_offset: 0.1
        optimizer_params:
          decouple: true
          use_bias_correction: False
          betas: [0.9, 0.99]
          weight_decay: 0.05
        skip_first_sample: true
        linear_timesteps: true
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true  
      sample:
        sampler: "flowmatch" 
        sample_every: 250 
        width: 1024
        height: 1024
        prompts:
          - "[trigger] a girl warrior with an axe."
          - "[trigger] a barbarian girl wearing crow glove."
          - "[trigger] a wizard holding a staff and blindfolded with bandages."
        neg: ""  
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20
meta:
  name: "[name]"
  version: '1.0'
```
설정파일 작성이 끝났다면 `/workspace/ai-toolkit/config` 디렉토리에 설정파일을 배치.

## 3. datasets 설정하기
위 설정에서 다음 옵션은 데이터세트 경로를 지정한다.
```yaml
      datasets:
        - folder_path: "/workspace/datasets"
```
사전 준비된 훈련이미지와 캡션(대응 되는 이미지 파일과 동일한 이름, 확장자 txt)을 `/workspace`(Volume Disk)로 전송하고 데이터세트 경로를 수정. Flux 캡션에 대한 정보는 다음을 참고하라.

- https://civitai.com/articles/6792/flux-style-captioning-differences-training-diary
- https://civitai.com/articles/7146/flux-style-captioning-differences-pt2-4-new-caption-tools-training-diary

저자는 [TagGUI](https://github.com/jhc13/taggui) 툴을 사용하여 캡션을 편집하였다. 여러 언어 모델을 지원하여 캡션 자동생성을 지원하고 일괄 편집이 용이하다. 
아래는 WD14 모델로 생성한 프롬프트가 적용된 모습이다. 
![](/images/245729e7-4d1e-4a72-89e5-ef412f90ac4a-image.png)


## 4. 훈련 시작
위 과정을 모두 완수하였다면 아래 커맨드로 훈련을 시작하면 된다.
```shell
root@some_id_blahblah:/workspace/ai-toolkit# python run.py config/some_config_filename.yaml
```

아래와 같이 출력된다면 정상적으로 시작된 것이다.
![](/images/7823d87f-04ad-4460-b864-fbf11fa79c85-image.png)

## 5. 훈련이 끝나면 인스턴스 종료
훈련이 끝나면, lora 파일과 스텝별 샘플은 `workspace/ai-toolkit/output/lora_name` 디렉토리에 저장되어 있다. 이를 로컬로 옮기고 사용이 끝난 인스턴스를 아래와 같이 Stop 버튼을 눌러 정지상태로 변경한다.
![](/images/e420d3f2-5366-4fbe-bf4c-1a9036bd57f3-image.png)

정지 상태에서도 Volume Disk 비용은 지출된다. 휴지통 버튼을 눌러 인스턴스를 삭제한다.
![](/images/6f24b370-8a78-4342-b92b-f2d3182d8364-image.png)


# Lora 훈련 결과 분석하기
lora 모델 파일은 설정된 스텝 수(`sample_every`)마다 `{lora name}_{step}.safetensors` 파일이름 규칙으로 생성되며 각 스텝 별로 생성된 샘플 이미지들은 `samples` 디렉토리에 있다.     
```shell
~/my_loras/pixel_sprite_lora_v3  ls
config.yaml                                pixel_sprite_lora_v3_000000750.safetensors pixel_sprite_lora_v3_000002000.safetensors
optimizer.pt                               pixel_sprite_lora_v3_000001000.safetensors pixel_sprite_lora_v3_000002250.safetensors
pixel_sprite_lora_v3.safetensors           pixel_sprite_lora_v3_000001250.safetensors pixel_sprite_lora_v3_000002500.safetensors
pixel_sprite_lora_v3_000000250.safetensors pixel_sprite_lora_v3_000001500.safetensors pixel_sprite_lora_v3_000002750.safetensors
pixel_sprite_lora_v3_000000500.safetensors pixel_sprite_lora_v3_000001750.safetensors samples
```

생성한 샘플 이미지들을 일목요연하게 보고 싶다면 다음 코드를 주피터노트북에서 실행하여라. 

```python
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
from collections import defaultdict

def visualize_training_progress(image_folder):
    # 이미지 파일 수집 및 정보 파싱
    images_info = []
    pattern = r'(\d+)__(\d+)_(\d+)\.jpg'
    
    for file in os.listdir(image_folder):
        if file.endswith('.jpg'):
            match = re.match(pattern, file)
            if match:
                time, step, number = map(int, match.groups())
                images_info.append({
                    'file': file,
                    'time': time,
                    'step': step,
                    'number': number,
                    'path': os.path.join(image_folder, file)
                })
    
    if not images_info:
        print("이미지를 찾을 수 없습니다.")
        return
    
    # step별로 이미지 그룹화
    step_groups = defaultdict(list)
    for info in images_info:
        step_groups[info['step']].append(info)
    
    # 각 step별로 정렬
    for step in step_groups:
        step_groups[step].sort(key=lambda x: x['number'])
    
    # 정렬된 step 목록
    sorted_steps = sorted(step_groups.keys())
    
    # 시각화 설정
    max_images_per_step = max(len(group) for group in step_groups.values())
    n_steps = len(sorted_steps)
    
    # 그리드 크기 설정 (이미지 크기 증가)
    plt.figure(figsize=(20, 2.5 * n_steps))  # 세로 크기 조정
    
    # 서브플롯 간격 조정
    plt.subplots_adjust(
        left=0.02,      # 왼쪽 여백
        right=0.98,     # 오른쪽 여백
        top=0.98,       # 위 여백
        bottom=0.02,    # 아래 여백
        hspace=0.1,     # 행간 간격
        wspace=0.1      # 열간 간격
    )
    
    # 각 스텝별로 이미지 표시
    for step_idx, step in enumerate(sorted_steps):
        images = step_groups[step]
        
        for img_idx, img_info in enumerate(images):
            # 서브플롯 생성
            ax = plt.subplot(n_steps, max_images_per_step, 
                           step_idx * max_images_per_step + img_idx + 1)
            
            # 이미지 로드 및 표시
            img = Image.open(img_info['path'])
            ax.imshow(np.array(img))
            ax.axis('off')
            
            # 첫 번째 이미지에만 스텝 정보 표시
            if img_idx == 0:
                ax.set_title(f"Step {step}\n#{img_info['number']}", 
                           loc='left', pad=5, fontsize=10)
            else:
                ax.set_title(f"#{img_info['number']}", 
                           loc='left', pad=5, fontsize=10)
    
    plt.suptitle("Lora Training Progress", fontsize=16, y=0.98)
    plt.show()
    
    # 통계 정보 출력
    print(f"\n총 {len(sorted_steps)} 스텝, {len(images_info)} 이미지 처리됨")
    print("\n스텝별 이미지 수:")
    for step in sorted_steps:
        print(f"Step {step}: {len(step_groups[step])} 이미지")

# 사용 예시
image_folder = "/somewhere_lorafile_path/samples"
visualize_training_progress(image_folder)
```

- 실행 결과
![](/images/3a13a837-dff9-48ac-b803-cbd9005cac19-image.png)

이제, 표를 보고 마음에 드는 step의 Lora 모델을 선택해 사용하면 된다.

# 소감 및 잡담
- RunPod로 시간과 비용 절감. 하지만 RTX4090 사고 싶음.
  ![](/images/4f486bc1-b01b-456f-aff1-cecb029732ac-image.png)

- 옵타마이저 Prodigy의 훈련 성과가 압도적이다. reddit에서 많이 언급된 이유가 있다.
- flux를 위한 img2vid가 없다.
- openpose controlnet은 artifacts 발생. strength를 0.6까지 내리면 되지만 포즈가 잘 적용되지 않는다.
  ![](/images/0f822fe7-b261-4b56-9ef2-6a4b7a162d5b-image.png)
- DWPose Estimator는 도트 스타일 인식이 안된다. 훈련데이터세트 편향이 사유.
