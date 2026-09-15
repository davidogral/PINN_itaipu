// simulacao/retrospectiva.rs — Simulação retrospectiva (Fase 6)

// Para cada dia histórico, compara a geração REAL com a curva de referência
// prevista pela PINN na vazão proxy observada.
//
// INTERPRETAÇÃO: a curva é uma referência física-informada treinada no
// período temporal definido em main.rs. Como Itaipu tem reservatório, os
// resíduos não devem ser lidos como ganho energético/financeiro nem como prova
// direta de eficiência operacional.

use std::error::Error;
use std::path::Path;

use crate::data::loader::Registro;
use crate::neural::forward::prever;
use crate::neural::loss::Fisica;
use crate::neural::network::Rede;

const HORAS_DIA: f64 = 24.0; // MWmed (potência média diária) × 24 h = MWh/dia

pub struct ResultadoDia {
    pub data: String,
    pub geracao_real: f64,  // MWmed
    pub geracao_otima: f64, // MWmed (nome histórico do CSV; curva de referência)
    pub delta_mwh: f64,     // (referência - real) × 24
}

pub struct Resumo {
    pub n_dias: usize,
    pub dias_com_ganho: usize,
    pub energia_real_mwh: f64,
    pub delta_mwh_liquido: f64,  // soma com sinal
    pub delta_mwh_positivo: f64, // soma de max(0, delta)
}

/// Roda a simulação dia a dia usando a PINN treinada.
pub fn simular(rede: &Rede, fis: &Fisica, registros: &[Registro]) -> (Vec<ResultadoDia>, Resumo) {
    let mut dias = Vec::with_capacity(registros.len());
    let (mut liquido, mut positivo, mut energia_real) = (0.0, 0.0, 0.0);
    let mut com_ganho = 0usize;

    for r in registros {
        let p_norm = prever(rede, &[r.vazao_norm])[0];
        let (_v, ger_otima_bruta) = fis.desnormalizar(r.vazao_norm, p_norm);
        let ger_otima = ger_otima_bruta.min(fis.p_max).max(0.0);
        let ger_real = r.geracao;

        let delta_mwh = (ger_otima - ger_real) * HORAS_DIA;
        liquido += delta_mwh;
        if delta_mwh > 0.0 {
            positivo += delta_mwh;
            com_ganho += 1;
        }
        energia_real += ger_real * HORAS_DIA;

        dias.push(ResultadoDia {
            data: r.data.clone(),
            geracao_real: ger_real,
            geracao_otima: ger_otima,
            delta_mwh,
        });
    }

    let resumo = Resumo {
        n_dias: registros.len(),
        dias_com_ganho: com_ganho,
        energia_real_mwh: energia_real,
        delta_mwh_liquido: liquido,
        delta_mwh_positivo: positivo,
    };
    (dias, resumo)
}

/// Salva o detalhe diário em CSV (para os gráficos da Fase 7).
pub fn salvar(dias: &[ResultadoDia], caminho: &Path) -> Result<(), Box<dyn Error>> {
    let mut w = csv::Writer::from_path(caminho)?;
    w.write_record([
        "data",
        "geracao_real_mwmed",
        "geracao_otima_mwmed",
        "delta_mwh",
    ])?;
    for d in dias {
        w.write_record(&[
            d.data.clone(),
            d.geracao_real.to_string(),
            d.geracao_otima.to_string(),
            d.delta_mwh.to_string(),
        ])?;
    }
    w.flush()?;
    Ok(())
}
