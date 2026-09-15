// neural/mod.rs — Módulo da rede neural

// Rede feedforward implementada do zero, sem libs de ML:
//   - `network`:   estrutura (camadas, pesos, vieses, init Xavier + PRNG).
//   - `forward`:   forward pass (tanh nas ocultas, linear na saída).
//   - `backward`:  backpropagation manual (+ gradient check nos testes).
//   - `optimizer`: Adam do zero.
//   - `loss`:      MSE (Loss_dados) — estende para física + contorno.

pub mod backward;
pub mod forward;
pub mod loss;
pub mod network;
pub mod optimizer;
