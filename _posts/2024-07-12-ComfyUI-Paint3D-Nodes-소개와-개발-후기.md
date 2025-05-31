---
title: "ComfyUI-Paint3D-Nodes 소개와 개발 후기"
description: "AI로 3D모델의 텍스쳐를 생성해보기."
date: 2024-07-12T11:23:55.977Z
tags: ["ComfyUI","Stable Diffusion","paint3d"]
thumbnail: /images/f32c26fa-da5f-4e6d-9d00-034ed643230d-image.png
---
> **주의! 본 게시글의 독자는 Stable Diffusion과 ComfyUI에 대한 사전지식을 가지고 있음을 가정하고 작성하였음**

**레포지토리**
[프로젝트 ComfyUI Paint3D Nodes](https://github.com/N3rd00d/ComfyUI-Paint3D-Nodes)

**참고**
- [프로젝트 Paint3D](https://github.com/OpenTexture/Paint3D)
- [Paint3D 논문](https://arxiv.org/pdf/2312.13913)

# 소개

ComfyUI-Paint3D-Nodes는 [Paint3D](https://arxiv.org/pdf/2312.13913)를 기반으로 만든 3D 모델 텍스쳐 인페인팅 ComfyUI 커스텀 노드이다.

![](/images/f32c26fa-da5f-4e6d-9d00-034ed643230d-image.png)

---

# 환경설정

```
|-----------|--------------|
| miniconda | 24.4.0       |
|-----------|--------------|
| python    | 3.10.14      |
|-----------|--------------|
| pytorch   | 2.1.0+cu121  |
|-----------|--------------|
| xformers  | 0.0.22.post4 |
|-----------|--------------|
| kaolin    | 0.15.0       |
|-----------|--------------|
```

**의존 라이브러리 설치**
```
pip install -r requirements.txt
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
```

참고
- https://github.com/OpenTexture/Paint3D
- https://github.com/NVIDIAGameWorks/kaolin

**이미 설치한 torch 버전이 kaolin을 지원하지 않는다면, 삭제하고 호환되는 버전으로 재설치** 
```
# 현재 설치된 torch 버전이 kaolin이 요구하는 버전과 불일치할 경우 torch를 삭제하고 재설치
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

**컨트롤넷**
- uvpos -  https://huggingface.co/GeorgeQi/Paint3d_UVPos_Control
- depth, inpaint - https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors 

---

# 워크플로우의 이해

본 프로젝트의 이해를 돕기 위해 작성된 샘플 워크플로우를 보면서 실행 동선과 커스텀 노드를 간결히 소개하고자 한다. Paint3D 노드는 모두 접두사 `3D_`를 가진다.

## 1. Standby 단계

모델들을 로드하고, 샘플링 그리고 인페인팅에 사용할 공통 프롬프트를 설정한다. `TrainConfig` 노드는 다음 단계에서 요구되는 모든 파라미터들을 사전에 설정 및 저장하고 `TrainChonfigPipe` 노드를 통해 공유하게 한다.

![](/images/fe44f89b-5034-48cc-b464-3332cd099506-image.png)

알베도 텍스처는 샘플링 과정에서 그려지는 광원과 그림자를 최대한 억제해서 생성해야 하기에 부정 프롬프트는 `strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses`와 같은 값을 설정해주는 것이 좋다. 개인적으로 효과는 미비하다고 판단되나, 없는 것보단 낫다고 생각된다.

이 프로젝트는 UV맵 전용 컨트롤넷에 의존적인 터라 SD1.5 버전만 지원된다.

## 2. 전/후면 샘플링 이미지 생성 및 투영

`LoadMeshModel` 노드는 `TrainComfig` 노드의 `mesh_file_path`로 설정된 경로의 obj 파일을 읽고 매시 정보를 메모리에 로드한다. `GenerateDepthImage` 노드는 매시 정보와 지정된 카메라 위치(0~25)에서 렌더되는 모델의 뎁스 이미지 2장을 생성한다. 이 이미지들을 1장으로 붙여 뎁스 컨트롤넷으로 사용되어 샘플링된다. 샘플링 이미지는 `Projection` 노드에 의해 모델의 텍스처로 투영된다.

![](/images/dd1c2fa9-baa7-46d4-a1e6-764acd445341-image.png)

## 3. 회전하면서 인페인팅 이미지 생성 및 투영

`GenerateInpaintMask` 노드는 지정된 카메라 위치에서 렌더되는 뷰포트에서 뎁스 이미지와 아직 인페인팅되지 않은 영역을 감지해 마스크 이미지를 생성한다. 이들 이미지는 각각 뎁스 컨트롤넷과 인페인트 컨트롤넷으로 사용되어 인페인팅된다. 인페인팅된 이미지는 마스크의 영역만큼 모델의 텍스처로 투영 적용된다.

![](/images/186107c8-20b4-42cc-b763-32d67aedf08a-image.png)

이러한 과정은 정면에서 45도/90도/135도 회전된 좌/우 이미지와 위/아래 이미지를 사용해 총 4번 인페인팅을 수행하게 된다. (논문에서는 정면에서 좌/우 45도 그리고 위/아래 이미지만 사용하지만, 다음 단계에서 UV Pos 컨트롤넷의 디테일을 올리고자 2단계를 더 추가함)

## 4. UV맵 인페인팅

`GenerateInpaintUVMap` 노드는 모델로부터 UV Pos 이미지를 생성한다. 이는 UV Pos 컨트롤넷의 이미지로 사용되어 텍스처를 **Light-less**(광원과 그림자를 제거)하게 만들어주게 한다. 마지막으로 UV Pos 맵을 마스크 이미지로 사용하여 투영의 경계 영역과 투영되지 않은 사각 영역을 채우는 인페인팅을 수행한다.

![](/images/76efbb4e-9136-4b3b-aecc-4416bab0a88b-image.png)

생성된 텍스처는 업스케일을 통해 2k 해상도로 맞추고, `SaveUVMapImage` 노드를 통해 텍스처를 png파일로 저장한다. 여기 과정까지 생성한 이미지들과 모델 그리고 텍스처 파일들은 obj 파일이 있는 위치에서 새로 생성된 `Paint3D` 디렉토리에서 확인할 수 있다.

---

# 커스텀 노드 개발 후기

뛰엄뛰엄 작업해서 코드 분석부터 지금까지 얼추 나흘 정도 소요된 것 같다. 개인적인 호기심과 도전정신으로 시작하였지만 comfyui와 torch, tensorflow 라이브러리를 따로 익힌 건 아니고, 그냥 직관(느낌적 느낌)으로 왠지 이렇게 코딩하면 될 거 같은데 하는 식으로 코딩을 하였기에 내가 모르는 실수나, 메모리 누수가 있을 수 있다.

생각보다 재미있는 작업이었고, 많은 사람들이 이 글을 보고 스테이블 디퓨젼에 많은 관심을 가졌으면 하는 바이다.
