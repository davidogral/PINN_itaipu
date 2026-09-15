// data/loader.rs — Leitura do dataset

// Lê `data/processed/dataset_final.csv` (gerado pelo Python na Fase 2)
// para dentro de structs Rust, usando `serde` + `csv`.

// Estrutura do CSV (separador vírgula, com cabeçalho):
//   data,vazao,ena_bruta,mes_sin,mes_cos,vazao_lag1,geracao_lag1,geracao,*_norm
//     - vazao        : vazão diária PORTO SAO JOSE (m³/s) [proxy de entrada]
//     - ena_bruta    : ENA da bacia Paraná (MWmed)
//     - *_lag1       : observações do dia anterior
//     - geracao      : geração média diária de Itaipu (MWmed) [alvo]
//     - *_norm       : versões normalizadas em [0,1] com min/max do treino

use serde::Deserialize;
use std::error::Error;
use std::path::Path;

/// Uma linha do dataset diário (2015–2024).
///
/// `*_norm` são lidos por `serde` mas só serão usados no treino (Fase 4+);
/// `allow(dead_code)` evita warning enquanto isso.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Registro {
    pub data: String,
    pub vazao: f64,
    #[serde(default)]
    pub ena_bruta: f64,
    #[serde(default)]
    pub mes_sin: f64,
    #[serde(default)]
    pub mes_cos: f64,
    #[serde(default)]
    pub vazao_lag1: f64,
    #[serde(default)]
    pub geracao_lag1: f64,
    pub geracao: f64,
    pub vazao_norm: f64,
    #[serde(default)]
    pub ena_bruta_norm: f64,
    #[serde(default)]
    pub vazao_lag1_norm: f64,
    #[serde(default)]
    pub geracao_lag1_norm: f64,
    pub geracao_norm: f64,
}

#[derive(Debug, Clone)]
pub struct AmostraModelo {
    pub entrada: Vec<f64>,
    pub alvo: f64,
    pub q_norm: f64,
}

/// Carrega todos os registros do CSV.
///
/// O `csv::Reader` usa vírgula como separador por padrão (compatível com o
/// arquivo exportado na Fase 2) e mapeia as colunas pelo cabeçalho via `serde`.
pub fn carregar_csv<P: AsRef<Path>>(caminho: P) -> Result<Vec<Registro>, Box<dyn Error>> {
    let mut leitor = csv::Reader::from_path(caminho)?;
    let mut registros = Vec::new();
    for resultado in leitor.deserialize() {
        let registro: Registro = resultado?;
        registros.push(registro);
    }
    Ok(registros)
}

/// Extrai os pares `(vazao, geracao)` — conveniência para a matemática/treino.
#[allow(dead_code)] // usado a partir da Fase 4 (treino)
pub fn pares_vazao_geracao(registros: &[Registro]) -> Vec<(f64, f64)> {
    registros.iter().map(|r| (r.vazao, r.geracao)).collect()
}

/// Extrai os pares normalizados `(vazao_norm, geracao_norm)`.
#[allow(dead_code)]
pub fn pares_normalizados(registros: &[Registro]) -> Vec<(f64, f64)> {
    registros
        .iter()
        .map(|r| (r.vazao_norm, r.geracao_norm))
        .collect()
}

/// Amostras da PINN univariada: entrada = vazão proxy normalizada.
pub fn amostras_vazao(registros: &[Registro]) -> Vec<AmostraModelo> {
    registros
        .iter()
        .map(|r| AmostraModelo {
            entrada: vec![r.vazao_norm],
            alvo: r.geracao_norm,
            q_norm: r.vazao_norm,
        })
        .collect()
}

/// Amostras multivariadas sem geração defasada: vazão, ENA, sazonalidade e vazão defasada.
///
/// A vazão normalizada continua na posição 0 para que o termo físico varie
/// apenas essa coordenada ao estimar dP/dQ por diferenças finitas.
/// `geracao_lag1` fica no dataset apenas para o baseline de persistência.
pub fn amostras_multivariadas(registros: &[Registro]) -> Vec<AmostraModelo> {
    registros
        .iter()
        .map(|r| AmostraModelo {
            entrada: vec![
                r.vazao_norm,
                r.ena_bruta_norm,
                r.mes_sin,
                r.mes_cos,
                r.vazao_lag1_norm,
            ],
            alvo: r.geracao_norm,
            q_norm: r.vazao_norm,
        })
        .collect()
}

/// Amostras hidrológicas: vazão, ENA, sazonalidade e vazão defasada.
///
/// Esta versão remove `geracao_lag1` para avaliar desempenho sem dependência
/// autorregressiva direta do alvo.
pub fn amostras_hidrologicas(registros: &[Registro]) -> Vec<AmostraModelo> {
    registros
        .iter()
        .map(|r| AmostraModelo {
            entrada: vec![
                r.vazao_norm,
                r.ena_bruta_norm,
                r.mes_sin,
                r.mes_cos,
                r.vazao_lag1_norm,
            ],
            alvo: r.geracao_norm,
            q_norm: r.vazao_norm,
        })
        .collect()
}

/// Faixa min/max de uma variável (lida de `normalizacao.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Faixa {
    pub min: f64,
    pub max: f64,
}

/// Parâmetros de normalização salvos pela Fase 2 (campos extras são ignorados).
#[derive(Debug, Clone, Deserialize)]
pub struct Normalizacao {
    pub vazao: Faixa,
    pub geracao: Faixa,
}

/// Carrega `normalizacao.json` (para des-normalizar resultados e montar a física).
pub fn carregar_normalizacao<P: AsRef<Path>>(caminho: P) -> Result<Normalizacao, Box<dyn Error>> {
    let texto = std::fs::read_to_string(caminho)?;
    let norm: Normalizacao = serde_json::from_str(&texto)?;
    Ok(norm)
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn extrai_pares() {
        let regs = vec![Registro {
            data: "2015-01-01".into(),
            vazao: 3871.3,
            ena_bruta: 0.0,
            mes_sin: 0.0,
            mes_cos: 1.0,
            vazao_lag1: 3871.3,
            geracao_lag1: 8682.2,
            geracao: 8682.2,
            vazao_norm: 0.09,
            ena_bruta_norm: 0.0,
            vazao_lag1_norm: 0.09,
            geracao_lag1_norm: 0.43,
            geracao_norm: 0.43,
        }];
        let pares = pares_vazao_geracao(&regs);
        assert_eq!(pares.len(), 1);
        assert_eq!(pares[0].0, 3871.3);
    }
}
