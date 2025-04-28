---
title: "유나이트 서울 2025 - Mobile native debugging and ANR 노트정리"
description: "유나이트 서울 2019 이후로 코로나가 터지고 유니티도 터지고 아무튼 간만에 개최되는 행사이다. 사람은 많았고 강의실은 더웠다."
date: 2025-04-28T14:59:32.401Z
tags: ["2025","ANR","Unity","debugging","mobile","unity seoul","이희소"]
---
유나이트 서울 2019 이후로 코로나가 터지고 유니티도 터지고 아무튼 간만에 개최되는 행사이다. 
사람은 많았고 강의실은 더웠다. 봄이었다.

# 유나이트 서울 2025 - Mobile native debugging and ANR 노트정리

## ANR 개요
- **ANR (Application Not Responding)**: 앱이 사용자 입력에 일정 시간 내 미응답 시 안드로이드 OS가 감지
- 일반적으로 입력 이벤트 5초 이상 무응답 시 발생 (OEM별 상이)
- Unity 앱에서 주로 입력 이벤트 처리 지연이 원인
- iOS의 "App Freezing"과 유사

## ANR 진단 도구 및 절차

### 1. 디바이스 로그 수집
- 안드로이드 디버깅 설정 (무선 디버깅, ADB 페어링)
- Android Logcat Package 설치
- 메모리/CPU 로그, Input, Systrace 활용

### 2. 버그 리포트
- 시스템 리소스, 스레드 상태, 메모리 정보 포함
- 분석 복잡성 높아 사전 학습 권장

### 3. Unity 앱 내 ANR 진단
- 타사 SDK 제거 → 점진적 추가 방식으로 원인 추적
- 빈번한 기기/OS 버전 중심 분석
- Unity 프로파일러는 메인 스레드 측정 한계 → 외부 도구 활용

## ANR 발생 위치별 원인 분석

### 1. 게임 코드 (C#)
- `libil2cpp.so` 스택 트레이스 노출 시 확인
- `OnApplicationPause` 등 과도한 작업
- 무한 루프/동기 네트워크/IO 작업 지양
- async/await, UnityTask, 스레드 분리 추천

### 2. Unity 엔진 코드
- `libunity.so`, `libmain.so` 관련 트레이스
- 엔진 레벨 문제 → Unity 포럼/버그 리포트 활용
- 최신 LTS 릴리즈 노트 확인 권장

### 3. 서드 파티 SDK
- 오래된 SDK, WebView/Chromium 기반 광고 SDK 등에서 발생
- SDK 업데이트, SDK 인덱스 확인, 공급자에 문의
- 바인더 호출은 메인 스레드에서 피해야 함

### 4. 안드로이드 시스템 라이브러리
- 개발자가 직접 해결 어려움
- 구글 개발자 문서, 로그 강화, 시스템 리소스 추적 등 간접 대응

## 실제 ANR 사례 및 대응 전략

### 1. WebView/Chromium 연동
- **원인**: 광고 콘텐츠 렌더링, Shader 미지원, 리소스 과다 사용
- **대응**: Web 디버깅, 광고 비활성화 테스트, 메모리 디버깅

### 2. 무한 루프 (OnApplicationPause)
- Unity가 일시 정지 이벤트 중 무한 루프 수행 → ANR 유발
- **해결**: Stopwatch 사용, IO 작업 분리, await 도입

### 3. UnityPlayer 클래스의 Display Update
- GPU 드라이버/해제 대기 문제 → 세마포어 4초 초과 대기
- **해결**: 대기 시간 최소화 전략, Unity 팀이 해결 방안 테스트 중

### 4. 메시지 큐 블로킹
- 백그라운드 중 수신 불가 메시지 큐 대기 → 복귀 시 ANR
- **대응**: 메시지 최소화, 로컬 DB 저장 후 복귀 처리

### 5. Binder Call 관련
- IPC 기반 처리 지연 → 메인 스레드 블로킹
- **해결**: 메인 스레드에서 호출 금지, SDK 릴리즈 노트 확인, 계측 도입

### 6. 메인 스레드 I/O
- IO 자체보다는 Unity/Android 간 동기화 문제로 발생
- **대응**: 락 최소화, IO/바인더를 별도 스레드로 분리

### 7. 네이티브 락 경쟁 (Mutex)
- GPU 렌더링/CPU 작업 중 Mutex 대기
- **해결**: 소스 기반 뮤텍스 위치 분석, 프로파일러로 락 감지, 그래픽 품질 하향

## 추천 툴 및 리포팅 방법
- Application Exit Info (Unity 6)
- Google StrictMode / adb scripts
- Firebase Test Lab / Pre-launch Reports
- Watchdog service 구현 (2~3초 타임아웃 기반)

## 결론
- ANR은 Unity 엔진, 게임 코드, 외부 SDK, 시스템 전반에서 발생 가능
- 정확한 스택 트레이스 분석, 디바이스/OS/버전별 통계 확인, 반복 테스트 및 프로파일링이 중요
- Unity 최신 기능 (await, UnityTask 등) 적극 활용 필요
