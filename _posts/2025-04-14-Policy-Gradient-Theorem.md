---
title: "Policy Gradient Theorem"
description: "Policy Gradient Theorem은 강화학습에서 정책 파라미터에 대한 목적 함수의 그래디언트를 계산 가능한 형태로 표현한다. 이를 통해 에이전트가 환경과 상호작용하며 얻는 보상을 최대화하는 정책을 최적화할 수 있다."
date: 2025-04-14T13:56:01.931Z
tags: ["PolicyGradient","강화학습","기계학습","딥러닝","인공지능","정책최적화"]
---
# Policy Gradient Theorem

Policy Gradient Theorem은 강화학습에서 정책 파라미터에 대한 목적 함수의 그래디언트를 계산 가능한 형태로 표현한다. 이를 통해 에이전트가 환경과 상호작용하며 얻는 보상을 최대화하는 정책을 최적화할 수 있다. 이 정리는 REINFORCE, Actor-Critic, PPO 같은 알고리즘의 수학적 기반이 된다.

## 정책 최적화의 목표

강화학습에서 우리는 에이전트가 환경과 상호작용하면서 받는 보상을 최대화하는 정책을 찾고자 한다. 정책은 파라미터 $\theta$로 표현되는 함수 $\pi_\theta(a|s)$로 정의된다. 이는 상태 $s$에서 행동 $a$를 선택할 확률을 나타낸다.

정책의 성능은 목적 함수 $J(\theta)$로 측정하며, 이는 다음과 같이 정의된다:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

이 식은 정책 $\pi_\theta$를 따라 생성된 궤적들의 기대 보상을 나타낸다. 여기서 $\tau$는 상태와 행동의 시퀀스이고, $R(\tau)$는 해당 궤적에서 얻은 총 보상이다.

정책 최적화의 목표는 $J(\theta)$를 최대화하는 파라미터 $\theta$를 찾는 것이다. 일반적으로 이를 위해 그래디언트 어센트 방법을 사용한다:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

하지만 여기서 실질적인 문제가 발생한다. $\nabla_\theta J(\theta)$를 어떻게 계산할 수 있을까? 보상 함수 $R(\tau)$는 환경에 의해 결정되며, 정책 파라미터 $\theta$에 직접적으로 의존하지 않기 때문에 단순히 미분할 수 없다.

## 정리의 수학적 표현
Policy Gradient Theorem은 이 문제에 대한 해결책을 제시한다. 이 정리는 목적 함수의 그래디언트를 다음과 같이 표현한다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

이 공식을 통해 정책의 그래디언트는 각 상태-행동 쌍의 가치($Q^{\pi_\theta}$)와 로그 확률의 그래디언트의 곱의 기대값으로 표현됨을 알 수 있다. 이는 그래디언트 계산을 실제로 구현 가능한 형태로 만들어준다.

## 수학적 유도: 단계별 접근

이 놀라운 결과가 어떻게 도출되는지 단계별로 살펴보자.

### 1단계: 목적 함수의 그래디언트 설정

먼저 목적 함수를 더 명시적인 형태로 표현해 보자:

$$
J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)
$$

이 식에서 $P(\tau;\theta)$는 정책 $\pi_\theta$를 따를 때 궤적 $\tau$가 발생할 확률이다. 이 식의 그래디언트는 다음과 같다:

$$
\nabla_\theta J(\theta) = \sum_{\tau} R(\tau) \nabla_\theta P(\tau;\theta)
$$

### 2단계: 로그 그래디언트 트릭의 적용

여기서 로그 미분의 성질을 활용하는 로그 그래디언트 트릭을 적용한다. 모든 양수 함수 $f(x)$에 대해 다음 관계가 성립한다:

$$
\nabla_x f(x) = f(x) \nabla_x \log f(x)
$$

이 관계가 성립하는 이유는 미적분학의 연쇄 법칙과 로그 함수의 미분 성질 때문이다. 로그 함수의 미분은 다음과 같다:

$$
\frac{d}{dx} \log f(x) = \frac{1}{f(x)} \cdot \frac{d}{dx}f(x)
$$

이 식을 $\nabla_x \log f(x)$에 대해 정리하면:

$$
\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}
$$

따라서:

$$
f(x) \nabla_x \log f(x) = f(x) \cdot \frac{\nabla_x f(x)}{f(x)} = \nabla_x f(x)
$$

이 트릭은 특히 확률 분포의 그래디언트를 계산할 때 매우 유용하다. 확률 계산은 복잡한 곱셈으로 이루어지는 경우가 많은데, 로그를 취하면 합으로 변환되어 미분이 간단해지기 때문이다.

이 성질을 $P(\tau;\theta)$에 적용하면:

$$
\nabla_\theta P(\tau;\theta) = P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)
$$

따라서 목적 함수의 그래디언트는 다음과 같이 변환된다:

$$
\nabla_\theta J(\theta) = \sum_{\tau} R(\tau) P(\tau;\theta) \nabla_\theta \log P(\tau;\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) \nabla_\theta \log P(\tau;\theta)]
$$

### 3단계: 궤적의 확률은 다음과 같이 분해할 수 있다:

$$
P(\tau;\theta) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)
$$

여기서 $P(s_0)$는 초기 상태 분포이고, $P(s_{t+1} \mid s_t, a_t)$는 환경의 전이 확률이다. 이 식의 로그를 취하면:

$$
\log P(\tau;\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)
$$

이 식에서 초기 상태 분포와 환경 전이 확률은 정책 파라미터 $\theta$와 무관하므로, 그래디언트를 취할 때 사라진다:

$$
\nabla_\theta \log P(\tau;\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

### 4단계: 최종 정리 도출

이제 앞서 얻은 결과를 조합하면:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

위 식에서 $R(\tau)$는 궤적 전체에서 얻는 총 보상$(r_0 + r_1 + ... + r_{T-1})$을 의미한다. 그러나 이 식은 각 시간 단계 t의 행동이 t 이전의 보상에도 영향을 주는 것처럼 계산한다. 실제로는 t시점의 행동은 t 이후의 보상에만 영향을 미친다. 따라서 각 시간 단계에서는 해당 시점부터 얻을 수 있는 기대 보상인 Q-함수를 사용하는 것이 더 정확하다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$


## 마무리
이 정리는 계산이 어려웠던 정책 그래디언트를 계산할 수 있는 형태로 변환했다는 점에서 중요하다. 주요 장점은 다음과 같다:

1. **실용적인 구현**: 몬테카를로 샘플링을 통해 그래디언트를 근사할 수 있다.
2. **모델 프리 접근**: 환경의 역학을 알 필요 없이 학습이 가능하다.


## 참고
- 로라 그래서, 와 룬 켕. (2022). *단단한 심층강화학습*. 제이펍.
- 노승은. (2020). 바닥부터 배우는 강화학습. 영진닷컴.
