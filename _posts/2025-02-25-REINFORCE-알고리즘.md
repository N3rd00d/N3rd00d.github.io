---
title: "REINFORCE 알고리즘"
description: "REINFORCE 알고리즘은 심층신경망의 파라미터를 조정하여 목적 함수를 최대화 하는 것이다. "
date: 2025-02-25T20:36:15.626Z
tags: ["PolicyGradient","reinforce","강화학습","기계학습","딥러닝","인공지능","정책최적화"]
---
# REINFORCE 알고리즘

REINFORCE 알고리즘은 몬테카를로 방법으로 수집한 궤적들의 리턴 값을 사용하여 정책 그래디언트를 확률적으로 근사하고, 이를 통해 심층신경망의 파라미터($\theta$)를 조정함으로써 정책 최적화를 달성한다.

## Gradient Ascent

### 목적 함수 $J(\theta)$
목적 함수 $J(\theta)$는 주어진 정책 $\pi_\theta (a \mid s)$의 성능을 평가한다. 

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

- $\tau \sim \pi_\theta$는 정책 $\pi_\theta$에 따라 생성된 궤적 $\tau$를 의미한다.
- $\mathbb{E}_{\tau \sim \pi_\theta}$는 정책 $\pi_\theta$에 따라 생성된 모든 가능한 궤적 $\tau$에 대한 기대값을 계산한다.
- $R(\tau)$는 특정 궤적 $\tau$의 누적 보상으로, $R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...$로 표현된다.

좋은 정책은 기대 보상을 최대화하는 정책이다. 목적 함수 $J(\theta)$는 __가능한 모든 궤적에 대한 기대 보상의 평균을 계산함으로써 정책의 성능을 수치화한다.__

정책 $\pi_\theta$는 주어진 상태 $s$에서 행동 $a$를 선택할 확률분포 $\mathbb{P}(a \mid s ; \theta)$를 의미한다, 기대값 표기를 제거하고 다음과 같이 표현할 수 있다.

$$
J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)
$$

- $P(\tau;\theta)$는 특정 궤적 $\tau$가 발생할 확률을 나타내며, $P(\tau;\theta) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)$로 계산된다.

### 목적함수의 그래디언트 $\nabla_\theta J(\theta)$

그래디언트 어센트(Gradient Ascent)는 정책의 파라미터 $\theta$를 최적화하여 목적 함수 $J(\theta)$를 최대화하는 기법이다. 이는 다음과 같은 업데이트 규칙을 따른다:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

여기서 $\alpha$는 학습률(learning rate)이다. 각 업데이트가 진행될수록 $J(\theta_{t+1})$은 $J(\theta_t)$보다 증가하게 되며, 이 과정을 반복하면 최적 정책의 파라미터 $\theta^*$에 수렴하게 된다. 우리의 핵심 과제는 __$\nabla_\theta J(\theta)$를 효과적으로 계산하는 것이다.__

$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

## Policy Gradient Theorem

그런데 여기서 문제가 발생한다. $R(\tau)$는 $\theta$에 대해 직접적으로 미분할 수 없다. 보상은 환경에서 주어지는 값으로, 정책 파라미터와 직접적인 함수 관계가 없기 때문이다. 

이 문제를 해결하기 위해 Policy Gradient Theorem를 사용할 수 있다. 이 정리는 목적 함수의 그래디언트를 다음과 같이 변환한다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

이 식은 $\theta$에 대해 미분 가능하며, 몬테카를로 샘플링을 통해 근사할 수 있는 형태로 변환될 수 있다. 이것이 REINFORCE 알고리즘의 핵심 아이디어이다.

## Policy Gradient Theorem 유도 과정

Policy Gradient Theorem의 수식이 어떻게 도출되는지 단계별로 살펴보자. 우리의 목표는 $\nabla_\theta J(\theta)$를 계산하는 것이다.

### 1. 목적 함수의 그래디언트 전개

먼저 목적 함수를 다시 살펴보면:

$$
J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)
$$

이 식의 그래디언트를 양변에 취하고:

$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau;\theta) R(\tau)
$$

합의 미분은 미분의 합과 같으므로:

$$
\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta [P(\tau;\theta) R(\tau)]
$$

여기서 $R(\tau)$는 $\theta$와 무관하므로:

$$
\nabla_\theta J(\theta) = \sum_{\tau} R(\tau) \nabla_\theta P(\tau;\theta)
$$

### 2. 로그 그래디언트 트릭(Log Gradient Trick) 적용

로그 미분의 성질을 활용하는 단계이다. 어떤 함수 $f(x)$에 대해 다음과 같은 관계가 성립한다:

$$
\nabla_x \log f(x) = \frac{\nabla_x f(x)}{f(x)}
$$

이 식을 변형하면:

$$
\nabla_x f(x) = f(x) \nabla_x \log f(x)
$$

이 성질을 $P(\tau;\theta)$에 적용하면 다음과 같다:

$$
\nabla_\theta P(\tau;\theta) = P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)
$$

이 트릭은 계산이 복잡한 확률 분포의 그래디언트를 더 다루기 쉬운 로그 확률의 그래디언트로 변환한다. 이를 적용하면:

$$
\nabla_\theta J(\theta) = \sum_{\tau} R(\tau) \nabla_\theta P(\tau;\theta) = \sum_{\tau} R(\tau) P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)
$$

이는 기대값 형태로 다시 쓸 수 있다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) \nabla_\theta \log P(\tau;\theta)]
$$

여기서 기대값 $E_{\tau \sim \pi_\theta}$는 정책 $\pi_\theta$를 따르는 모든 가능한 궤적들의 가중 평균을 계산한다.

### 3. $\log P(\tau;\theta)$의 그래디언트 전개

궤적의 확률은 다음과 같이 표현된다:

$$
P(\tau;\theta) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)
$$

여기서 로그의 중요한 성질을 이용할 수 있다. 로그 함수는 곱셈을 덧셈으로 변환하는 성질이 있다:

$$
\log(a \cdot b) = \log(a) + \log(b)
$$

이 성질을 이용하면 곱의 형태로 표현된 복잡한 식을 합의 형태로 변환할 수 있어 미분 계산이 훨씬 간단해진다. 위 궤적 확률식에 로그를 취하면:

$$
\log P(\tau;\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t)
$$

여기서 $P(s_0)$와 $P(s_{t+1} \mid s_t, a_t)$는 환경 역학으로 $\theta$와 무관하다. 따라서 그래디언트는:

$$
\nabla_\theta \log P(\tau;\theta) = \nabla_\theta \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$


### 4. 최종 정리

이제 위 결과를 적용하면:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

이 식에서 전체 궤적의 보상 합 $R(\tau)$는 각 단계별 상태-행동 가치인 $Q^{\pi_\theta}(s_t, a_t)$로 대체할 수 있다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

이것이 Policy Gradient Theorem의 최종 형태이다.

## REINFORCE 알고리즘

REINFORCE 알고리즘은 Policy Gradient Theorem을 실제 구현한 방법이다. 이 알고리즘은 이론적인 $Q^{\pi_\theta}(s_t, a_t)$ 대신 실제 경험의 리턴 값을 사용한다.

### 몬테카를로 샘플링을 통한 근사

Policy Gradient Theorem의 수식은 기대값 형태로 표현된다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

현실에서는 이 기대값을 정확히 계산하기 어렵다. 모든 상태와 행동을 고려해야 하고, 정확한 $Q^{\pi_\theta}$ 값도 알 수 없기 때문이다. REINFORCE 알고리즘은 이 문제를 몬테카를로 샘플링으로 해결한다.

몬테카를로 샘플링은 무작위 샘플을 추출하여 통계적 추정을 수행하는 기법이다. 현재 정책 $\pi_\theta$로 여러 궤적을 생성하고, 각 궤적의 리턴 값으로 그래디언트를 추정한다.

시간 $t$에서의 리턴 $G_t$는 해당 시점부터 궤적 끝까지 받은 보상의 합이다:
$$
G_t = \sum_{k=t}^{T} r_k
$$

이 $G_t$는 $Q^{\pi_\theta}(s_t, a_t)$의 샘플 추정치다. 단일 궤적은 노이즈가 있지만, 다수 궤적이 모이면 평균은 정확도는 높아진다. 몬테카를로 추정치는 다음과 같다:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i-1} G_t^i \nabla_\theta \log \pi_\theta (a_t^i \mid s_t^i)
$$

큰 수의 법칙에 따라, 샘플 수 $N$이 증가할수록 이 추정치는 실제 기대값에 가까워진다. 충분한 궤적을 모으면 추정 그래디언트는 정책 경사의 참값에 근접한다.

REINFORCE 알고리즘은 이 추정치로 정책 파라미터를 점진적으로 업데이트한다. 각 업데이트는 노이즈를 포함하지만, 장기적으로 최적 정책에 도달한다.