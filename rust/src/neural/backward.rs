// neural/backward.rs — Backpropagation

// Calcula os gradientes da loss (MSE) em relação a todos os pesos e
// vieses, sem diferenciação automática.
//
// Saída linear  => δ_L = dL/dz_L = 2·(pred - alvo)        (por amostra)
// Ocultas tanh  => δ_l = (Wᵀ_{l+1} · δ_{l+1}) ⊙ (1 - a_l²)
//   pois tanh'(z) = 1 - tanh(z)² = 1 - a².
// grad_W_l = δ_l · a_{l-1}ᵀ ;  grad_b_l = δ_l
//
// Gradientes são acumulados sobre o lote e divididos por N (=> grad do MSE).
// Validado por gradient check (diferenças finitas) no teste.

use crate::neural::forward::forward;
use crate::neural::network::Rede;

/// Gradientes, com a mesma forma da rede.
#[derive(Clone)]
pub struct Gradientes {
    pub d_pesos: Vec<Vec<Vec<f64>>>,
    pub d_vieses: Vec<Vec<f64>>,
}

impl Gradientes {
    pub fn zeros(rede: &Rede) -> Self {
        let d_pesos = rede
            .camadas
            .iter()
            .map(|c| c.pesos.iter().map(|linha| vec![0.0; linha.len()]).collect())
            .collect();
        let d_vieses = rede
            .camadas
            .iter()
            .map(|c| vec![0.0; c.vieses.len()])
            .collect();
        Gradientes { d_pesos, d_vieses }
    }
}

/// Acumula em `grads` os gradientes de UMA amostra; retorna a perda (sem média).
fn retropropagar_amostra(
    rede: &Rede,
    entrada: &[f64],
    alvo: &[f64],
    grads: &mut Gradientes,
) -> f64 {
    let l = rede.num_camadas();
    let cache = forward(rede, entrada);
    let saida = cache.ativacoes.last().unwrap();

    let perda: f64 = saida.iter().zip(alvo).map(|(p, t)| (p - t) * (p - t)).sum();

    // δ da camada de saída (linear): dL/dz = 2(pred - alvo)
    let mut delta: Vec<f64> = saida.iter().zip(alvo).map(|(p, t)| 2.0 * (p - t)).collect();

    for i in (0..l).rev() {
        let a_prev = &cache.ativacoes[i];
        // gradientes da camada i
        for j in 0..rede.camadas[i].pesos.len() {
            grads.d_vieses[i][j] += delta[j];
            for k in 0..a_prev.len() {
                grads.d_pesos[i][j][k] += delta[j] * a_prev[k];
            }
        }
        // propaga δ para a camada anterior (oculta, tanh)
        if i > 0 {
            let largura_prev = rede.camadas[i - 1].vieses.len();
            let mut novo = vec![0.0; largura_prev];
            for k in 0..largura_prev {
                let mut soma = 0.0;
                for j in 0..rede.camadas[i].pesos.len() {
                    soma += rede.camadas[i].pesos[j][k] * delta[j];
                }
                // tanh'(z_{i-1}) = 1 - a², com a = saída tanh da camada i-1
                let a = cache.ativacoes[i][k];
                novo[k] = soma * (1.0 - a * a);
            }
            delta = novo;
        }
    }

    perda
}

/// Backprop "genérico": acumula gradientes dado o δ da camada de SAÍDA
/// (= dL/dz_L) já calculado pelo chamador. É a primitiva usada pela PINN
/// Onde cada termo da loss — dados, física, contorno — vira um δ de
/// saída que se soma. Para a saída linear, δ_saida = dL/da_L.
pub fn acumular_saida(rede: &Rede, entrada: &[f64], delta_saida: &[f64], grads: &mut Gradientes) {
    let l = rede.num_camadas();
    let cache = forward(rede, entrada);
    let mut delta = delta_saida.to_vec();

    for i in (0..l).rev() {
        let a_prev = &cache.ativacoes[i];
        for j in 0..rede.camadas[i].pesos.len() {
            grads.d_vieses[i][j] += delta[j];
            for k in 0..a_prev.len() {
                grads.d_pesos[i][j][k] += delta[j] * a_prev[k];
            }
        }
        if i > 0 {
            let largura_prev = rede.camadas[i - 1].vieses.len();
            let mut novo = vec![0.0; largura_prev];
            for k in 0..largura_prev {
                let mut soma = 0.0;
                for j in 0..rede.camadas[i].pesos.len() {
                    soma += rede.camadas[i].pesos[j][k] * delta[j];
                }
                let a = cache.ativacoes[i][k];
                novo[k] = soma * (1.0 - a * a);
            }
            delta = novo;
        }
    }
}

/// Gradientes médios e MSE médio sobre um lote de `(entrada, alvo)`.
pub fn gradientes_lote(rede: &Rede, lote: &[(Vec<f64>, Vec<f64>)]) -> (Gradientes, f64) {
    let mut grads = Gradientes::zeros(rede);
    let mut perda_total = 0.0;

    for (entrada, alvo) in lote {
        perda_total += retropropagar_amostra(rede, entrada, alvo, &mut grads);
    }

    let n = lote.len() as f64;
    for i in 0..grads.d_pesos.len() {
        for j in 0..grads.d_pesos[i].len() {
            grads.d_vieses[i][j] /= n;
            for k in 0..grads.d_pesos[i][j].len() {
                grads.d_pesos[i][j][k] /= n;
            }
        }
    }

    (grads, perda_total / n)
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::neural::forward::prever;

    /// MSE médio do lote (para o gradient check numérico).
    fn perda_lote(rede: &Rede, lote: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        let n = lote.len() as f64;
        lote.iter()
            .map(|(e, t)| {
                let p = prever(rede, e);
                p.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()
            })
            .sum::<f64>()
            / n
    }

    #[test]
    fn gradient_check() {
        // Rede pequena e lote pequeno; compara backprop com diferenças finitas.
        let mut rede = Rede::nova(&[1, 3, 1], 123);
        let lote = vec![
            (vec![0.3], vec![0.5]),
            (vec![-0.7], vec![0.2]),
            (vec![0.9], vec![-0.1]),
        ];
        let (grads, _) = gradientes_lote(&rede, &lote);

        let eps = 1e-6;
        let mut max_diff = 0.0_f64;
        for i in 0..rede.camadas.len() {
            for j in 0..rede.camadas[i].pesos.len() {
                for k in 0..rede.camadas[i].pesos[j].len() {
                    let orig = rede.camadas[i].pesos[j][k];
                    rede.camadas[i].pesos[j][k] = orig + eps;
                    let mais = perda_lote(&rede, &lote);
                    rede.camadas[i].pesos[j][k] = orig - eps;
                    let menos = perda_lote(&rede, &lote);
                    rede.camadas[i].pesos[j][k] = orig;

                    let numerico = (mais - menos) / (2.0 * eps);
                    let analitico = grads.d_pesos[i][j][k];
                    max_diff = max_diff.max((numerico - analitico).abs());
                }
            }
        }
        assert!(
            max_diff < 1e-6,
            "gradient check falhou: max_diff={max_diff:.2e}"
        );
    }
}
