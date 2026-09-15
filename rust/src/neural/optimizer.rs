// neural/optimizer.rs — Otimizador Adam

// Atualiza os parâmetros da rede com Adam (Kingma & Ba, 2015):
//   m = β1·m + (1-β1)·g
//   v = β2·v + (1-β2)·g²
//   m̂ = m/(1-β1ᵗ) ;  v̂ = v/(1-β2ᵗ)
//   θ ← θ - lr · m̂ / (√v̂ + ε)
//
// Estado (m, v) por parâmetro. Padrões: β1=0.9, β2=0.999, ε=1e-8.

use crate::neural::backward::Gradientes;
use crate::neural::network::Rede;

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: i32,
    m_pesos: Vec<Vec<Vec<f64>>>,
    v_pesos: Vec<Vec<Vec<f64>>>,
    m_vieses: Vec<Vec<f64>>,
    v_vieses: Vec<Vec<f64>>,
}

impl Adam {
    pub fn novo(rede: &Rede, lr: f64) -> Self {
        let z = Gradientes::zeros(rede);
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_pesos: z.d_pesos.clone(),
            v_pesos: z.d_pesos,
            m_vieses: z.d_vieses.clone(),
            v_vieses: z.d_vieses,
        }
    }

    /// Um passo de atualização sobre todos os parâmetros da rede.
    pub fn passo(&mut self, rede: &mut Rede, grads: &Gradientes) {
        self.t += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let corr1 = 1.0 - b1.powi(self.t);
        let corr2 = 1.0 - b2.powi(self.t);

        for i in 0..rede.camadas.len() {
            for j in 0..rede.camadas[i].pesos.len() {
                // viés
                let g = grads.d_vieses[i][j];
                self.m_vieses[i][j] = b1 * self.m_vieses[i][j] + (1.0 - b1) * g;
                self.v_vieses[i][j] = b2 * self.v_vieses[i][j] + (1.0 - b2) * g * g;
                let mh = self.m_vieses[i][j] / corr1;
                let vh = self.v_vieses[i][j] / corr2;
                rede.camadas[i].vieses[j] -= self.lr * mh / (vh.sqrt() + self.eps);

                // pesos
                for k in 0..rede.camadas[i].pesos[j].len() {
                    let g = grads.d_pesos[i][j][k];
                    self.m_pesos[i][j][k] = b1 * self.m_pesos[i][j][k] + (1.0 - b1) * g;
                    self.v_pesos[i][j][k] = b2 * self.v_pesos[i][j][k] + (1.0 - b2) * g * g;
                    let mh = self.m_pesos[i][j][k] / corr1;
                    let vh = self.v_pesos[i][j][k] / corr2;
                    rede.camadas[i].pesos[j][k] -= self.lr * mh / (vh.sqrt() + self.eps);
                }
            }
        }
    }
}
