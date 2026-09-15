// math/mod.rs — Módulo de matemática base (Fase 3)

// Métodos numéricos clássicos implementados do zero, usados como
// baseline de comparação com a PINN:
//   - `interpolacao`:   interpolação quadrática (Lagrange/Newton).
//   - `newton_raphson`: raízes/extremos 1D.

pub mod interpolacao;
pub mod newton_raphson;
