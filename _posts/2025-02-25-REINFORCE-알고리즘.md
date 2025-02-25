---
title: "REINFORCE 알고리즘"
description: "REINFORCE 알고리즘은 심층신경망의 파라미터를 조정하여 목적 함수를 최대화 하는 것이다. "
date: 2025-02-25T20:36:15.626Z
tags: ["PolicyGradient","reinforce","강화학습","기계학습","딥러닝","인공지능","정책최적화"]
---
# REINFORCE 알고리즘

REINFORCE 알고리즘은 __심층신경망의 파라미터($\theta$)를 조정하여 목적 함수를 최대화__ 하는 것이다. 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} R_t(\tau) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

## Gradient Ascent

### 목적 함수 $J(\theta)$
목적 함수 $J(\theta)$는 주어진 정책 $\pi_\theta (a \mid s)$의 성능을 평가한다. 이 함수의 값을 증가시키는 방향으로 $\theta$를 업데이트한다면, 정책 $\pi_\theta (a \mid s)$의 보상(리턴)은 점차 향상될 것이다.

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

여기서:
- $\tau \sim \pi_\theta$는 정책 $\pi_\theta$에 따라 생성된 궤적 $\tau$를 의미한다.
- $\mathbb{E}_{\tau \sim \pi_\theta}$는 정책 $\pi_\theta$에 따라 생성된 모든 가능한 궤적 $\tau$에 대한 기대값을 계산한다.
- $R(\tau)$는 특정 궤적 $\tau$의 누적 보상으로, $R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...$로 표현된다.

따라서 좋은 정책이란 기대 보상을 최대화하는 정책이다. 목적 함수 $J(\theta)$는 __가능한 모든 궤적에 대한 기대 보상의 평균을 계산함으로써 정책의 성능을 수치화한다.__

정책 $\pi_\theta$는 주어진 상태 $s$에서 행동 $a$를 선택할 확률분포 $\mathbb{P}(a \mid s ; \theta)$를 출력하므로, 기대값 표기를 제거하고 다음과 같이 명시적으로 표현할 수도 있다:

$$
J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)
$$

여기서:
- $P(\tau;\theta)$는 특정 에피소드 $\tau$가 발생할 확률을 나타내며, $P(\tau;\theta) = P(s_0) \prod_{t=0}^{T-1} P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)$로 계산된다.

### 목적함수의 그래디언트 $\nabla_\theta J(\theta)$

그래디언트 어센트(Gradient Ascent)는 정책의 파라미터 $\theta$를 최적화하여 목적 함수 $J(\theta)$를 최대화하는 기법이다. 이는 다음과 같은 업데이트 규칙을 따른다:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
$$

여기서 $\alpha$는 학습률(learning rate)이다. 각 업데이트가 진행될수록 $J(\theta_{t+1})$은 $J(\theta_t)$보다 증가하게 되며, 이 과정을 반복하면 최적 정책의 파라미터 $\theta^*$에 수렴하게 된다. 따라서 우리의 핵심 과제는 $\nabla_\theta J(\theta)$를 효과적으로 계산하는 것이다.

$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

## Policy Gradient Theorem

그런데 여기서 문제가 발생한다. $R(\tau)$는 $\theta$에 대해 직접적으로 미분할 수 없다. 보상은 환경에서 주어지는 값으로, 정책 파라미터와 직접적인 함수 관계가 없기 때문이다. 

이 문제를 해결하기 위해 Policy Gradient Theorem이 등장했다. 이 정리는 목적 함수의 그래디언트를 다음과 같이 변환한다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} Q(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

이 식은 $\theta$에 대해 미분 가능하며, 몬테카를로 샘플링을 통해 근사할 수 있는 형태로 변환되었다. 이것이 REINFORCE 알고리즘의 핵심 아이디어이다.

## Policy Gradient Theorem 유도 과정

Policy Gradient Theorem의 수식이 어떻게 도출되는지 단계별로 살펴보자. 우리의 목표는 $\nabla_\theta J(\theta)$를 계산하는 것이다.

### 1. 목적 함수의 그래디언트 전개

먼저 목적 함수를 다시 살펴보면:

$$
J(\theta) = \sum_{\tau} P(\tau;\theta) R(\tau)
$$

이 식의 그래디언트를 구하면:

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

이 트릭은 계산이 복잡한 확률 분포의 그래디언트를 더 다루기 쉬운 로그 확률의 그래디언트로 변환해주기 때문에 머신러닝에서 자주 사용된다. 이를 적용하면:

$$
\nabla_\theta J(\theta) = \sum_{\tau} R(\tau) \nabla_\theta P(\tau;\theta) = \sum_{\tau} R(\tau) P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)
$$

이는 기대값 형태로 다시 쓸 수 있다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) \nabla_\theta \log P(\tau;\theta)]
$$

여기서 $\mathbb{E}_{\tau \sim \pi_\theta}$는 정책 $\pi_\theta$에 따라 생성된 모든 궤적에 대한 기대값을 의미한다.

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

실제로는 미래 행동이 과거 보상에 영향을 줄 수 없다는 인과성(causality)을 고려하여, 각 시점 $t$에서는 해당 시점 이후의 보상만을 고려해야 한다. 따라서:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T} r_{t'} \right]
$$

위 식은 다시 쓰면:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

여기서 $G_t = \sum_{t'=t}^{T} r_{t'}$는 시간 $t$부터의 누적 보상이다.

Q-함수의 정의에 따라 $Q(s_t, a_t) = \mathbb{E}[G_t \mid s_t, a_t]$이므로, 최종적으로:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} Q(s_t, a_t) \nabla_\theta \log \pi_\theta (a_t \mid s_t) \right]
$$

이것이 Policy Gradient Theorem의 최종 형태이다. 

## REINFORCE 알고리즘

REINFORCE 알고리즘은 Policy Gradient Theorem을 몬테카를로 방식으로 구현한 것이다. 현실에서는 Q-함수의 정확한 값을 알기 어렵다. 따라서 REINFORCE 알고리즘에서는, Q-함수 대신 경험에서 얻은 리턴(return) 값 $G_t$를 사용한다. 이는 몬테카를로 방식으로 Q-함수를 추정하는 것과 같다.

$G_t$는 시간 $t$ 이후에 받은 모든 보상의 합으로, $G_t = \sum_{k=t}^{T} r_k$로 정의된다. 이를 사용하면 Policy Gradient는 다음과 같이 근사된다:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i} G_t^i \nabla_\theta \log \pi_\theta (a_t^i \mid s_t^i)
$$

이 식은 여러 에피소드에서 얻은 경험의 평균을 사용하여 그래디언트를 추정하는 것이다. 하지만 실제 구현에서는 대개 계산 효율성을 위해 단일 에피소드나 미니배치만을 사용하여 파라미터를 갱신한다. 이는 정확한 기대값을 계산하는 대신 확률적 근사를 사용하는 것으로, 흔히 확률적 경사 상승법(stochastic gradient ascent)이라 부른다. 단일 에피소드에 대한 그래디언트 추정은 다음과 같다:

$$
\nabla_\theta J(\theta) \approx \hat{g} := \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta (a_t \mid s_t)
$$

이렇게 계산된 그래디언트 $\hat{g}$를 사용하여 정책 파라미터 $\theta$를 다음과 같이 업데이트한다:

$$
\theta \leftarrow \theta + \alpha \hat{g}
$$

이것이 REINFORCE 알고리즘의 핵심이다. 글 시작에서 언급한 것처럼, REINFORCE 알고리즘은 심층신경망의 파라미터($\theta$)를 조정하여 목적 함수를 최대화하는 알고리즘으로, Policy Gradient Theorem과 몬테카를로 샘플링을 결합하여 실제 구현 가능한 형태로 만든 것이다.


## REINFORCE 알고리즘 의사코드

REINFORCE 알고리즘을 의사코드로 표현하면 다음과 같다:

```
초기화:
    정책 파라미터 θ를 임의의 값으로 초기화
    학습률 α 설정

while 종료조건을 만족하지 않음 do:
    # 에피소드 샘플링
    π_θ를 따라 에피소드 τ = {s₀, a₀, r₁, s₁, a₁, r₂, ..., sₙ} 생성
    
    # 리턴 계산
    G[0...n-1] ← 배열 생성 및 0으로 초기화
    for t = 0 to n-1 do:
        for k = t to n-1 do:
            G[t] ← G[t] + γ^(k-t) * r_{k+1}  # t부터의 감가된 보상 합
        end for
    end for
    
    # 그래디언트 추정 및 파라미터 업데이트
    g ← 0
    for t = 0 to n-1 do:
        g ← g + G[t] * ∇_θ log π_θ(a_t|s_t)
    end for
    
    θ ← θ + α * g
end while
```

이 알고리즘은 다음과 같은 단계로 동작한다:

1. 현재 정책 π_θ를 사용하여 완전한 에피소드를 생성한다.
2. 각 시간 단계 t에서의 리턴 G[t]를 계산한다 (시간 t부터의 모든 보상에 감가율을 적용한 합).
3. 모든 시간 단계에 대한 정책 그래디언트를 계산하고 합산한다.
4. 계산된 그래디언트를 사용하여 정책 파라미터 θ를 업데이트한다.
5. 위 과정을 수렴할 때까지 반복한다.
