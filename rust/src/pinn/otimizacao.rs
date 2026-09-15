// pinn/otimizacao.rs — Curva de referência pós-treino

// Com a PINN treinada, para cada nível de vazão proxy calcula a geração de
// referência (saída da rede, limitada à capacidade) e a produtividade relativa
// (MW por m³/s proxy).
//
// O "joelho" da curva é o primeiro ponto da referência em que a geração se
// aproxima da capacidade. O q_sat usado no termo físico é definido a priori
// em loss.rs. Salva curva_otima.csv (nome histórico do pipeline).

use std::error::Error;
use std::path::Path;

use crate::neural::forward::prever;
use crate::neural::loss::Fisica;
use crate::neural::network::Rede;

/// Um ponto da curva de referência, em unidades físicas.
pub struct PontoOtimo {
    pub vazao: f64,         // m³/s
    pub geracao_otima: f64, // MWmed (nome histórico; geração de referência)
    pub eficiencia: f64,    // MWmed por (m³/s proxy)
}

/// Gera a curva de referência varrendo a faixa de vazões observadas.
pub fn curva_otima(rede: &Rede, fis: &Fisica, n_pontos: usize) -> Vec<PontoOtimo> {
    let mut curva = Vec::with_capacity(n_pontos);
    for i in 0..n_pontos {
        let q_norm = i as f64 / (n_pontos - 1) as f64; // [0,1]
        let p_norm = prever(rede, &[q_norm])[0];
        let (vazao, geracao) = fis.desnormalizar(q_norm, p_norm);
        // limita fisicamente à capacidade instalada
        let geracao_otima = geracao.min(fis.p_max).max(0.0);
        let eficiencia = if vazao > 1.0 {
            geracao_otima / vazao
        } else {
            0.0
        };
        curva.push(PontoOtimo {
            vazao,
            geracao_otima,
            eficiencia,
        });
    }
    curva
}

/// Vazão de saturação definida pelo termo físico, em unidades físicas.
pub fn vazao_saturacao(fis: &Fisica) -> f64 {
    fis.q_min + fis.q_sat_norm * (fis.q_max - fis.q_min)
}

/// Salva a curva de referência em CSV para a análise (Fase 6) e os gráficos.
pub fn salvar_curva(curva: &[PontoOtimo], caminho: &Path) -> Result<(), Box<dyn Error>> {
    let mut w = csv::Writer::from_path(caminho)?;
    w.write_record(["vazao", "geracao_otima", "eficiencia"])?;
    for p in curva {
        w.write_record(&[
            p.vazao.to_string(),
            p.geracao_otima.to_string(),
            p.eficiencia.to_string(),
        ])?;
    }
    w.flush()?;
    Ok(())
}
