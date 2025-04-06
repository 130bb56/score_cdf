Using data.txt, we can get the below result

![Screenshot (545)](https://github.com/130bb56/score_cdf/assets/125135262/f8e15aaf-4b96-4182-b57b-53ca47488e31)

## HPMC: High-Performance MNIST Classification

### 1. 개요
HPMC는 PyTorch 대비 최대 8배 빠른 학습 속도를 달성한 C++/CUDA 기반 MLP 구현 프로젝트입니다. cuBLAS나 CUTLASS와 같은 고수준 라이브러리 없이 End-to-End 학습 파이프라인을 직접 구축하였으며, GEMM 최적화, 커널 퓨전, 공유 메모리 타일링 등 다양한 CUDA 최적화 전략을 적용하였습니다. 모델은 MNIST 손글씨 숫자 분류를 위한 2-hidden-layer MLP(784→320→160→10)입니다.

---

### 2. 문제 정의 및 동기
기존 PyTorch 기반 MLP 학습은 학습 속도 및 GPU 리소스 활용률 측면에서 최적화 여지가 많습니다. 또한, PyTorch가 내부적으로 어떤 방식으로 연산을 최적화하고 있는지, 이를 저수준 CUDA에서 직접 재현하고 분석함으로써 ML 시스템 최적화에 대한 인사이트를 얻고자 본 프로젝트를 시작하였습니다. 특히, 실서비스에서 중요한 latency 단축과 GPU 자원 효율성을 직접 개선해보고자 했습니다.

---

### 3. 접근 방식 및 해결 전략

#### ▣ 초기 접근
- PyTorch baseline 코드 작성 (`nn.Linear`, `SGD`, `CrossEntropy` 기반)
- CUDA 포팅: `forward`, `relu`, `softmax` 각각 별도 구현하여 End-to-End 학습
- 모델 구조: 784 → 320 → 160 → 10
- 하이퍼파라미터 설정: `batch_size = 64`, `lr = 0.03`
- Tensor Shape 및 수식 명시적으로 계산하여 각 커널 설계

#### ▣ 중기 전략
- `forward + relu`, `forward + softmax` 커널 퓨전 적용 → latency 1600ms → 300ms로 급감
- 기존 `nvprof`는 deprecated, `nsight compute`는 과다한 metric 제공 → CUDA Event 기반 Custom Profiler(`profiler.cuh`) 직접 구현
- profiler instance를 global 선언하고 각 kernel 호출 전후에 타이머 삽입, 실행 후 평균 latency 측정 및 출력 스크립트 작성

#### ▣ 후기 전략
- GEMM 최적화:
  - shared memory tiling
  - bank conflict 방지를 위한 padding 적용
  - loop unrolling (#pragma unroll)
- `cross_entropy` 커널:
  - 초기에는 16개의 thread로 reduction 수행 → 10개의 class에 대해 FP 오차 발생
  - sequential loop + unrolling 방식으로 수정 → PyTorch와 동일한 정확도 확보
- `update_layer` 커널:
  - bias 업데이트 과정에서 동일 column에 여러 thread 접근 발생
  - atomicAdd() 사용하여 race condition 해결

---

### 4. 벤치마킹 및 실험 결과

#### ▣ 실험 설정
- batch_size = 64, epoch = 30, block_size = 16, TILE_SIZE = 16
- 환경: RTX 4060 Laptop GPU

#### ▣ 요약 테이블
| 항목 | PyTorch | CUDA | 개선 |
|------|---------|------|------|
| Time / Epoch | 1961ms | 218ms | 8~9배 빠름 |
| GPU Utilization | 34% | 64% | +30%p |
| Validation Accuracy | 97.78% | 97.84% | 동등 수준 |
| Memory Usage | 145MiB | 126MiB | 감소 |

---

### 5. 느낀점 및 한계점
- 단순한 구현을 넘어서 **시스템 수준의 ML 성능 병목을 식별하고 최적화**하는 과정을 직접 수행
- PyTorch 내부의 추상화를 걷어내고, CUDA에서 연산 단위별 latency, shape, memory access pattern을 분석함으로써 **ML 시스템 아키텍처 전반에 대한 실전적 경험** 확보
- 한계점: MLP 구조는 CNN/LLM 등 복잡한 구조에 비해 단순하여 일반화에는 제한이 있지만, CUDA 최적화 기본기 학습에는 매우 효과적이었음
- 추후 계획: TensorCore `wmma::mma_sync`, online softmax 분리, binary dataset preprocessing, CLI 하이퍼파라미터 설정 등

---

### 📎 Appendix

#### ▣ 구성 전략 및 컴파일 최적화
- 커널 그리드/블록 구성 시, 성능과 코드 명확성을 위해 컴파일 타임과 런타임 올림 계산 선택적 사용
- **컴파일 타임 상수**: `constexpr int _ceil(int a, int b)`를 정의하여 런타임 오버헤드 없이 `ceil(a / b)` 계산
- **런타임 변수**: 불필요한 float-to-double 변환을 피하기 위해 `std::ceil()` 대신 `std::ceilf()` 사용
- 컴파일러 옵션: `nvcc -O3 -NDEBUG -Xptxas=-O3 -arch=sm_89 -maxrregcount=64`

#### ▣ 디버깅 전략
- 두 가지 에러 검사 매크로 제공:
  ```cpp
  #define CHECK_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
  #define CHECK_KERNEL_ERROR() { cudaKernelAssert(__FILE__, __LINE__); }
  ```
- 동기식(`cudaAssert()`) 및 비동기식(`cudaKernelAssert()`) 오류 검사
- `--debug` 플래그로 전체 커널 수준 동기 디버깅 활성화 가능
  ```cpp
  inline void cudaKernelAssert(const char *file, const int line, bool abort = true) {
      if (debug) CHECK_ERROR(cudaDeviceSynchronize());
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "cudaKernelAssert(): %s\n[%s: %d]\n\n", cudaGetErrorString(err), file, line);
          if (abort) exit(err);
      }
  }
  ```

#### ▣ 실행 방법
```bash
git clone https://github.com/130bb56/hpmc.git
cd hpmc/data 
chmod +x ./download_mnist_dataset.sh && ./download_mnist_dataset.sh
cd ../cuda && make && ./a
```

#### ▣ 향후 개선 계획
- 조건부 분기 제거 (예: `if (row < height && col < width)`)로 warp divergence 감소
- `forward_softmax`를 별도의 `forward`와 `softmax` 커널로 분리하여 동기화 오버헤드 제거
- CSV 데이터 로딩을 바이너리 전처리 데이터셋으로 대체하여 CPU I/O 병목(~2초) 감소
- FP16 TensorCore `wmma::mma_sync` 활용
- 메모리 결합 전치(memory-coalesced transpose) 및 공유 메모리 정렬로 GEMM 최적화
- 루프 언롤링된 벡터화 메모리 로드(`reinterpret_cast<float4*>`) 사용
- CLI 인자를 통한 `batch_size`, `lr`, 레이어 차원 설정 지원

#### ▣ E2E 수식
MLP의 순전파 전체 수식:
\[ \hat{y} = \text{softmax}( (\text{ReLU}( (\text{ReLU}(X W^{(1)} + b^{(1)})) W^{(2)} + b^{(2)} ) W^{(3)} + b^{(3)} ) \]

#### ▣ Tensor Shape
| Layer | Input Shape | Weight Shape | Output Shape | Activation |
|-------|--------------|---------------|----------------|------------|
| Layer1 | (64, 784) | (784, 320) | (64, 320) | ReLU |
| Layer2 | (64, 320) | (320, 160) | (64, 160) | ReLU |
| Layer3 | (64, 160) | (160, 10) | (64, 10) | Softmax |

#### ▣ 수식 유도
- Softmax:
  \[ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \]
- Cross Entropy:
  \[ L = -\sum_i y_i \log(\hat{y}_i) \]
- Backward (Cross Entropy + Softmax):
  \[ \frac{\partial L}{\partial z_i} = \hat{y}_i - y_i \]
- Backpropagation:
  \[ \frac{\partial L}{\partial z^{(k)}} = \left( \frac{\partial L}{\partial z^{(k+1)}} W^{(k+1)T} \right) \odot 1(z^{(k)} > 0) \]

#### ▣ CUDA 커널 설명 (한글)
- `forward_relu`: 블록 및 그리드 구성은 `(output_dim / block_size, batch_size / block_size)` 2D 형태로 구성. 각 스레드 블록 내에서 shared memory를 이용한 타일 기반 GEMM(`X @ W + b`)을 수행하고, 이후 ReLU 활성화를 결합한 단일 커널로 처리함. padding을 통해 bank conflict를 완화하고, 반복문은 `#pragma unroll`로 전개함.
- `forward_softmax`: `forward_relu`와 동일한 grid/block 설정을 사용하며, softmax 연산을 포함하는 forward 경로를 하나의 커널로 결합. shared memory를 통한 타일링 최적화를 동일하게 적용.
- `z_grad`: 역전파 과정에서의 gradient 계산 커널. 다음 레이어의 gradient와 가중치 전치행렬의 곱(`dz^(k+1) @ W^(k+1)^T`)을 수행하고, 여기에 ReLU의 도함수(`Z > 0`)를 elementwise 곱하여 `dz^(k)`를 구함. 반복문 전개 및 shared memory tiling 적용.
- `cross_entropy`: softmax 출력을 기반으로 cross-entropy loss를 계산하는 커널. 최초에는 warp-level reduction(`__shfl_down_sync`)을 사용했으나, FP 정밀도 문제로 인해 sequential loop 방식으로 변경함. `WIDTH=10`을 define하여 loop unrolling 적용.
- `cross_entropy_softmax_grad`: `dz = y_hat - y` 형태로 softmax + cross entropy의 gradient를 직접 계산하는 커널. 연산량을 줄이기 위해 조건 분기 제거.
- `update_layer`: gradient를 기반으로 weight와 bias를 동시에 업데이트하는 커널. 동일 column에 대해 여러 thread가 접근하는 문제로 인해 bias는 `atomicAdd()`를 통해 업데이트함. `1/B` 스케일링은 loss 계산 시 제외하고 최종 업데이트 단계에서만 적용.
