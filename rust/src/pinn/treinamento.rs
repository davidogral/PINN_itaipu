// pinn/treinamento.rs — Loop de treinamento da PINN (Fase 5)

// Monta o gradiente combinado de Loss_total = λ_d·dados + λ_f·física + λ_c·contorno
// reaproveitando o backprop genérico `acumular_saida`: cada termo vira um δ de
// saída que se soma. Loga os três termos por época e salva o histórico.
//
// Todos os termos são resíduos sobre a SAÍDA da rede (ou sua derivada via
// diferenças finitas), então o backprop é reutilizado tal qual.

use std::error::Error;
use std::path::Path;

use crate::data::loader::AmostraModelo;
use crate::neural::backward::{acumular_saida, Gradientes};
use crate::neural::forward::prever;
use crate::neural::loss::{Fisica, Lambdas, Termos};
use crate::neural::network::Rede;
use crate::neural::optimizer::Adam;

/// Passo das diferenças finitas para dP/dQ no espaço normalizado.
const H_FD: f64 = 1e-3;

/// Calcula os gradientes da Loss_total e os valores dos três termos.
///
/// `dados` contêm a entrada normalizada da rede, o alvo normalizado e a vazão
/// normalizada. Em modelos multivariados, a física perturba somente a vazão
/// (posição 0 da entrada), mantendo as demais covariáveis fixas.
pub fn gradientes_pinn(
    rede: &Rede,
    dados: &[AmostraModelo],
    fis: &Fisica,
    lam: &Lambdas,
) -> (Gradientes, Termos) {
    let mut grads = Gradientes::zeros(rede);
    let n = dados.len() as f64;
    let inv = 1.0 / n;

    let (mut soma_dados, mut soma_edp, mut soma_cont) = (0.0, 0.0, 0.0);

    for amostra in dados {
        let q = amostra.q_norm;
        let g = amostra.alvo;
        let entrada = &amostra.entrada;

        // --- Loss_dados + teto de capacidade (mesmo input q) ---
        let nq = prever(rede, entrada)[0];
        let excesso = (nq - 1.0).max(0.0); // P_norm acima da capacidade (=1.0)
        let delta_dados = lam.dados * 2.0 * (nq - g);
        let delta_teto = lam.contorno * 2.0 * excesso;
        acumular_saida(
            rede,
            entrada,
            &[(delta_dados + delta_teto) * inv],
            &mut grads,
        );
        soma_dados += (nq - g) * (nq - g);
        soma_cont += excesso * excesso;

        // --- Loss_fisica: inclinação por diferenças finitas ---
        let mut entrada_mais = entrada.clone();
        let mut entrada_menos = entrada.clone();
        entrada_mais[0] = q + H_FD;
        entrada_menos[0] = q - H_FD;
        let np = prever(rede, &entrada_mais)[0];
        let nm = prever(rede, &entrada_menos)[0];
        let s = (np - nm) / (2.0 * H_FD);
        let residuo = s - fis.inclinacao_alvo(q);
        // dLoss_fisica/dN(q±h) = ±(s - alvo)/h ; aplica em cada lado
        let seed = lam.edp * residuo / H_FD * inv;
        acumular_saida(rede, &entrada_mais, &[seed], &mut grads);
        acumular_saida(rede, &entrada_menos, &[-seed], &mut grads);
        soma_edp += residuo * residuo;
    }

    // --- Loss_contorno: âncora P(Q=0)=0 (ponto único, peso próprio) ---
    let mut entrada_contorno = vec![0.0; rede.arquitetura[0]];
    entrada_contorno[0] = fis.q0_norm;
    let nb = prever(rede, &entrada_contorno)[0];
    let resid_b = nb - fis.p0_norm;
    acumular_saida(
        rede,
        &entrada_contorno,
        &[lam.contorno * 2.0 * resid_b],
        &mut grads,
    );
    soma_cont += resid_b * resid_b;

    let termos = Termos {
        dados: soma_dados / n,
        edp: soma_edp / n,
        contorno: soma_cont / n,
        total: lam.dados * soma_dados / n + lam.edp * soma_edp / n + lam.contorno * soma_cont / n,
    };
    (grads, termos)
}

/// Treina a PINN por `epocas` e devolve o histórico de termos por época.
/// `verboso` controla o log a cada 200 épocas (desligado nas rodadas de ablação).
pub fn treinar(
    rede: &mut Rede,
    dados: &[AmostraModelo],
    fis: &Fisica,
    lam: &Lambdas,
    lr: f64,
    epocas: usize,
    verboso: bool,
) -> Vec<Termos> {
    let mut adam = Adam::novo(rede, lr);
    let mut historico = Vec::with_capacity(epocas + 1);

    for epoca in 0..=epocas {
        let (grads, termos) = gradientes_pinn(rede, dados, fis, lam);
        historico.push(termos);
        if verboso && (epoca % 200 == 0 || epoca == epocas) {
            println!(
                "[pinn] época {epoca:>4} | total={:.5} | dados={:.5} fisica={:.5} contorno={:.5}",
                termos.total, termos.dados, termos.edp, termos.contorno
            );
        }
        if epoca < epocas {
            adam.passo(rede, &grads);
        }
    }

    historico
}

/// Salva o histórico de loss em CSV (para a Fase 7).
pub fn salvar_historico(historico: &[Termos], caminho: &Path) -> Result<(), Box<dyn Error>> {
    let mut w = csv::Writer::from_path(caminho)?;
    w.write_record([
        "epoca",
        "loss_total",
        "loss_dados",
        "loss_fisica",
        "loss_contorno",
    ])?;
    for (epoca, t) in historico.iter().enumerate() {
        w.write_record(&[
            epoca.to_string(),
            t.total.to_string(),
            t.dados.to_string(),
            t.edp.to_string(),
            t.contorno.to_string(),
        ])?;
    }
    w.flush()?;
    Ok(())
}
