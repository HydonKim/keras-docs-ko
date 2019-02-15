
## 활성화 함수 사용법

순방향 레이어에 대해 `Activation` 레이어, 혹은 `activation` 아규먼트로 활성화 함수를 활용할 수 있습니다:

```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

위 코드는 다음과 같이 작성할 수 있습니다:

```python
model.add(Dense(64, activation='tanh'))
```

TensorFlow/Theano/CNTK의 함수를 활성화 함수로 사용할 수 있습니다:

```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
```

## 활성화 함수 예시

### softmax


```python
keras.activations.softmax(x, axis=-1)
```


Softmax activation function.

__입력__

- __x__: 텐서.
- __axis__: 정수, softmax 정규화가 적용된 축.

__출력__

softmax를 통해 변환된 텐서.

__예외 처리__

- __ValueError__: 예) `dim(x) == 1`.
    
----

### elu


```python
keras.activations.elu(x, alpha=1.0)
```


Exponential linear unit.

__입력__

- __x__: 텐서.
- __alpha__: 스칼라, 음수부의 기울기.

__출력__

The exponential linear activation: `x > 0`일 때 `x`, 
`x < 0`일 때 `alpha * (exp(x)-1)`를 반환.

__참고 자료__

- [Fast and Accurate Deep Network Learning by Exponential
   Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    
----

### selu


```python
keras.activations.selu(x)
```


Scaled Exponential Linear Unit (SELU).

SELU는 사전에 정의된 alpha와 scale을 이용해 
`scale * elu(x, alpha)`와 같이 표현할 수 있습니다.
가중치가 올바르게 초기화되고 입력의 수가 충분히 크다면 
`alpha`와 `scale`은 연속된 두 레이어의 입력값의 평균와 분산이 보존되도록 선택됩니다.

__입력__

- __x__: 활성화 함수가 계산 가능한 형태의 텐서 혹은 변수.

__출력__

   The scaled exponential unit activation: `scale * elu(x, alpha)`를 반환.

__추가__

- "lecun_normal" 초기화와 함께 사용할 수 있습니다.
- "AlphaDropout"과 같은 드롭아웃 기법과 함께 사용할 수 있습니다.

__참고 자료__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    
----

### softplus


```python
keras.activations.softplus(x)
```


Softplus activation function.

__입력__

- __x__: 텐서.

__출력__

The softplus activation: `log(exp(x) + 1)`를 반환.
    
----

### softsign


```python
keras.activations.softsign(x)
```


Softsign activation function.

__입력__

- __x__: 텐서.

__출력__

The softsign activation: `x / (abs(x) + 1)`를 반환.
    
----

### relu


```python
keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
```


Rectified Linear Unit.

기본값으로 `max(x, 0)`를 반환합니다.

위의 경우가 아니라면
 `x >= max_value`일 때 `f(x) = max_value`,
`threshold <= x < max_value` 일 때 `f(x) = x`,
그 외의 경우 `f(x) = alpha * (x - threshold)`를 반환합니다.

__입력__

- __x__: 텐서.
- __alpha__: 소수. 음수부의 기울기(기본값: 0).
- __max_value__: 소수. Saturation 한계값.
- __threshold__: 소수. thresholded activation의 한계값.

__출력__

텐서.
    
----

### tanh


```python
keras.activations.tanh(x)
```


Hyperbolic tangent activation function.

----

### sigmoid


```python
keras.activations.sigmoid(x)
```


Sigmoid activation function.

----

### hard_sigmoid


```python
keras.activations.hard_sigmoid(x)
```


Hard sigmoid activation function.

sigmoid activation보다 빠름.

__입력__

- __x__: 텐서.

__출력__

Hard sigmoid activation:

- `x < -2.5`일 때 `0`
- `x > 2.5`일 때 `1` 
- `-2.5 <= x <= 2.5`일 때 `0.2 * x + 0.5`를 반환.

----

### exponential


```python
keras.activations.exponential(x)
```


Exponential (base e) activation function.

----

### linear


```python
keras.activations.linear(x)
```


Linear (i.e. identity) activation function.


## On "Advanced Activations"

간단한 TensorFlow/Theano/CNTK 함수(예. learnable activations, which maintain a state)보다 복잡한 활성화 함수는 [Advanced Activation layers](layers/advanced-activations.md)로 활용할 수 있으며 `keras.layers.advanced_activations` 모듈에서 찾아볼 수 있습니다. 이는 `PReLU`와 `LeakyReLU`를 포함합니다.
