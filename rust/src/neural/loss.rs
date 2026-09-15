// neural/loss.rs — Função de perda

// Loss_dados = MSE (erro quadrático médio).
// Loss_total = λ_d·Loss_dados + λ_f·Loss_fisica + λ_c·Loss_contorno
//         (ver docs/decisoes_tecnicas.md §5 e §8).
//
// Física embutida (forma DIFERENCIAL — escolha registrada):
//   P = ρ·g·Q·H·η  =>  dP/dQ = ρ·g·H·η = k  (constante) no regime produtivo.
//   A vazão usada é a de Porto São José, uma proxy de disponibilidade hídrica
//   a montante, não a vazão turbinada. Por isso impomos a INCLINAÇÃO como
//   regularização física, não o valor direto de P. A inclinação-alvo cai a 0
//   acima da vazão de saturação definida a priori => captura o platô.

/// Erro quadrático médio entre previsão e alvo (vetores de mesmo tamanho).
#[allow(dead_code)]
pub fn mse(pred: &[f64], alvo: &[f64]) -> f64 {
    debug_assert_eq!(pred.len(), alvo.len());
    let n = pred.len() as f64;
    pred.iter()
        .zip(alvo)
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / n
}

/// MSE médio sobre um lote de previsões escalares `(pred, alvo)`.
#[allow(dead_code)]
pub fn mse_lote(predicoes: &[(f64, f64)]) -> f64 {
    let n = predicoes.len() as f64;
    predicoes
        .iter()
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / n
}

/// Derivada do MSE em relação a cada previsão: `2/n · (pred - alvo)`.
#[allow(dead_code)]
pub fn d_mse(pred: &[f64], alvo: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;
    pred.iter()
        .zip(alvo)
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}

// Física da PINN

/// Constantes físicas da equação de potência hidráulica de Itaipu.
pub const RHO: f64 = 1000.0; // densidade da água (kg/m³)
pub const G: f64 = 9.81; // gravidade (m/s²)
pub const H_NET: f64 = 118.0; // queda líquida nominal de Itaipu (m)
pub const ETA: f64 = 0.90; // eficiência turbina-gerador

/// k = ρ·g·H·η, em MW por (m³/s)  (≈ 1.042).
pub fn k_fisico_mw_por_m3s() -> f64 {
    RHO * G * H_NET * ETA / 1.0e6
}

/// Pesos de cada termo da loss
#[derive(Debug, Clone, Copy)]
pub struct Lambdas {
    pub dados: f64,
    pub edp: f64,
    pub contorno: f64,
}

/// Valor de cada termo da loss numa época (para logging/ablação).
#[derive(Debug, Clone, Copy, Default)]
pub struct Termos {
    pub dados: f64,
    pub edp: f64,
    pub contorno: f64,
    pub total: f64,
}

/// Parâmetros físicos já convertidos para o espaço NORMALIZADO da rede.
///
/// A rede opera em `[0,1]` (min-max). Guardamos as faixas para des-normalizar
/// e os alvos físicos em coordenadas normalizadas.
#[derive(Debug, Clone, Copy)]
pub struct Fisica {
    pub q_min: f64,
    pub q_max: f64,
    pub p_min: f64,
    pub p_max: f64,
    /// Inclinação física no espaço normalizado: k·(Δq/Δp).
    pub k_norm: f64,
    /// Vazão de saturação (normalizada): onde P=k·Q atinge a capacidade.
    /// Este limiar é definido a priori pelo termo físico, não aprendido.
    pub q_sat_norm: f64,
    /// Coordenadas normalizadas do ponto de contorno P(Q=0)=0.
    pub q0_norm: f64,
    pub p0_norm: f64,
}

impl Fisica {
    pub fn nova(q_min: f64, q_max: f64, p_min: f64, p_max: f64) -> Self {
        let dq = q_max - q_min;
        let dp = p_max - p_min;
        let k = k_fisico_mw_por_m3s();
        let q_sat = p_max / k; // vazão física onde k·Q = capacidade
        Fisica {
            q_min,
            q_max,
            p_min,
            p_max,
            k_norm: k * dq / dp,
            q_sat_norm: (q_sat - q_min) / dq,
            q0_norm: -q_min / dq, // normalize(0)
            p0_norm: -p_min / dp, // normalize(0)
        }
    }

    /// Inclinação-alvo `dP_norm/dQ_norm` no ponto `q_norm`:
    /// `k_norm` no regime produtivo, `0` após a saturação.
    pub fn inclinacao_alvo(&self, q_norm: f64) -> f64 {
        if q_norm < self.q_sat_norm {
            self.k_norm
        } else {
            0.0
        }
    }

    /// Des-normaliza vazão e geração de volta às unidades físicas.
    pub fn desnormalizar(&self, q_norm: f64, p_norm: f64) -> (f64, f64) {
        (
            self.q_min + q_norm * (self.q_max - self.q_min),
            self.p_min + p_norm * (self.p_max - self.p_min),
        )
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn mse_zero_quando_igual() {
        assert_eq!(mse(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn mse_conhecido() {
        // erros 1 e 3 => (1+9)/2 = 5
        assert!((mse(&[2.0, 5.0], &[1.0, 2.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn k_fisico_na_faixa_esperada() {
        // ρgHη/1e6 ≈ 1.042 MW por m³/s
        assert!((k_fisico_mw_por_m3s() - 1.0418).abs() < 1e-3);
    }

    #[test]
    fn saturacao_dentro_do_dominio() {
        // Faixas reais (normalizacao.json): q[2181,20433], p[4657,13946]
        let f = Fisica::nova(2181.0, 20433.0, 4657.0, 13946.0);
        assert!(
            f.q_sat_norm > 0.0 && f.q_sat_norm < 1.0,
            "q_sat_norm={}",
            f.q_sat_norm
        );
        // antes da saturação a inclinação é k_norm; depois, 0
        assert_eq!(f.inclinacao_alvo(0.1), f.k_norm);
        assert_eq!(f.inclinacao_alvo(0.99), 0.0);
    }
}
