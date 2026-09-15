// pinn/mod.rs — Módulo da PINN

// Orquestra a Physics-Informed Neural Network sobre o módulo `neural`:
//   - `treinamento`: loop de treino com a loss combinada (dados+física+contorno).
//   - `otimizacao`:  curva de referência por nível de vazão proxy (pós-treino).

pub mod otimizacao;
pub mod treinamento;
