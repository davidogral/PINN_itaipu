// experimentos.rs -- Rodadas adicionais para investigar o plato preditivo.
//
// A ideia aqui e manter a Fase 5 intacta e adicionar uma avaliacao honesta de
// memoria temporal, alvos residuais/anomalias e fatores efetivos aprendidos.

use std::path::Path;
use std::time::Instant;

use crate::data::loader::Registro;
use crate::ml::arvores::{AmostraArvore, ParametrosFloresta, RandomForestRegressor};
use crate::neural::backward::gradientes_lote;
use crate::neural::forward::prever;
use crate::neural::loss::{k_fisico_mw_por_m3s, Fisica};
use crate::neural::network::Rede;
use crate::neural::optimizer::Adam;

const DIR_OUTPUTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/outputs");
const MAX_LAG: usize = 30;
const EPOCAS_MLP_EXPLORACAO: usize = 1200;
const LR_MLP_EXPLORACAO: f64 = 0.01;

const FEATURES_HIDRO_BASICO: &[&str] = &["mes_sin", "mes_cos", "q_lag1", "ena_lag1"];

const FEATURES_HIDRO_LONGO: &[&str] = &[
    "mes_sin",
    "mes_cos",
    "q_lag1",
    "q_lag2",
    "q_lag3",
    "q_lag7",
    "q_lag14",
    "q_lag30",
    "q_ma3",
    "q_ma7",
    "q_ma14",
    "q_ma30",
    "dq_lag1",
    "dq_lag7",
    "ena_lag1",
    "ena_lag2",
    "ena_lag3",
    "ena_lag7",
    "ena_lag14",
    "ena_lag30",
    "ena_ma3",
    "ena_ma7",
    "ena_ma14",
    "ena_ma30",
    "dena_lag1",
    "dena_lag7",
];

const FEATURES_OPERACIONAL: &[&str] = &[
    "mes_sin",
    "mes_cos",
    "q_lag1",
    "q_lag2",
    "q_lag3",
    "q_lag7",
    "q_lag14",
    "q_lag30",
    "q_ma3",
    "q_ma7",
    "q_ma14",
    "q_ma30",
    "dq_lag1",
    "dq_lag7",
    "ena_lag1",
    "ena_lag2",
    "ena_lag3",
    "ena_lag7",
    "ena_lag14",
    "ena_lag30",
    "ena_ma3",
    "ena_ma7",
    "ena_ma14",
    "ena_ma30",
    "dena_lag1",
    "dena_lag7",
    "p_lag1",
    "p_lag2",
    "p_lag7",
    "p_ma3",
    "p_ma7",
    "dp_lag1",
    "dp_lag7",
];

#[derive(Clone)]
struct LinhaTemporal {
    data: String,
    mes: usize,
    geracao: f64,
    geracao_lag1: f64,
    q_lag1: f64,
    q_ma7: f64,
    q_ma30: f64,
    hidro_basico: Vec<f64>,
    hidro_longo: Vec<f64>,
    operacional: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GrupoFeatures {
    HidroBasico,
    HidroLongo,
    Operacional,
}

impl GrupoFeatures {
    fn todos() -> [Self; 3] {
        [
            GrupoFeatures::HidroBasico,
            GrupoFeatures::HidroLongo,
            GrupoFeatures::Operacional,
        ]
    }

    fn nome(self) -> &'static str {
        match self {
            GrupoFeatures::HidroBasico => "hidro_lag1_basico",
            GrupoFeatures::HidroLongo => "hidro_lags_medias_30d",
            GrupoFeatures::Operacional => "operacional_lags_com_geracao",
        }
    }

    fn nomes_features(self) -> &'static [&'static str] {
        match self {
            GrupoFeatures::HidroBasico => FEATURES_HIDRO_BASICO,
            GrupoFeatures::HidroLongo => FEATURES_HIDRO_LONGO,
            GrupoFeatures::Operacional => FEATURES_OPERACIONAL,
        }
    }

    fn entrada(self, linha: &LinhaTemporal) -> &[f64] {
        match self {
            GrupoFeatures::HidroBasico => &linha.hidro_basico,
            GrupoFeatures::HidroLongo => &linha.hidro_longo,
            GrupoFeatures::Operacional => &linha.operacional,
        }
    }
}

#[derive(Clone)]
struct Escalador {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl Escalador {
    fn ajustar(linhas: &[LinhaTemporal], grupo: GrupoFeatures) -> Self {
        let n_features = grupo.nomes_features().len();
        let mut min = vec![f64::INFINITY; n_features];
        let mut max = vec![f64::NEG_INFINITY; n_features];

        for linha in linhas {
            for (i, valor) in grupo.entrada(linha).iter().enumerate() {
                min[i] = min[i].min(*valor);
                max[i] = max[i].max(*valor);
            }
        }

        Escalador { min, max }
    }

    fn transformar(&self, entrada: &[f64]) -> Vec<f64> {
        entrada
            .iter()
            .enumerate()
            .map(|(i, valor)| {
                let d = self.max[i] - self.min[i];
                if d.abs() < 1e-12 {
                    0.0
                } else {
                    (valor - self.min[i]) / d
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct ResultadoExperimento {
    modelo: String,
    formulacao: &'static str,
    grupo_features: String,
    periodo: &'static str,
    n: usize,
    rmse_mwmed: f64,
    mae_mwmed: f64,
    r2: f64,
    bias_mwmed: f64,
    tempo_treino_s: f64,
    observacao: String,
}

#[derive(Debug, Clone)]
struct PesoLinear {
    modelo: String,
    formulacao: &'static str,
    grupo_features: String,
    feature: String,
    peso: f64,
}

#[derive(Debug, Clone)]
struct FatorEfetivo {
    variavel: &'static str,
    intercepto: f64,
    k_efetivo_mw_por_m3s: f64,
    k_fisico_mw_por_m3s: f64,
    razao_k_efetivo_k_fisico: f64,
    rmse_teste_mwmed: f64,
    mae_teste_mwmed: f64,
    r2_teste: f64,
}

#[derive(Debug, Clone)]
struct ResultadoMovelOperacional {
    ano_teste: i32,
    modelo: String,
    formulacao: &'static str,
    n_treino: usize,
    n_teste: usize,
    rmse_mwmed: f64,
    mae_mwmed: f64,
    r2: f64,
    bias_mwmed: f64,
    melhoria_rmse_vs_persistencia_pct: f64,
    tempo_treino_s: f64,
    observacao: String,
}

#[derive(Clone)]
struct PerfilMensal {
    media: [f64; 13],
    delta: [f64; 13],
    fallback_media: f64,
    fallback_delta: f64,
}

impl PerfilMensal {
    fn ajustar(linhas: &[LinhaTemporal]) -> Self {
        let fallback_media = linhas.iter().map(|l| l.geracao).sum::<f64>() / linhas.len() as f64;
        let fallback_delta = linhas
            .iter()
            .map(|l| l.geracao - l.geracao_lag1)
            .sum::<f64>()
            / linhas.len() as f64;

        let mut soma_media = [0.0; 13];
        let mut soma_delta = [0.0; 13];
        let mut n = [0usize; 13];
        for linha in linhas {
            soma_media[linha.mes] += linha.geracao;
            soma_delta[linha.mes] += linha.geracao - linha.geracao_lag1;
            n[linha.mes] += 1;
        }

        let mut media = [fallback_media; 13];
        let mut delta = [fallback_delta; 13];
        for mes in 1..=12 {
            if n[mes] > 0 {
                media[mes] = soma_media[mes] / n[mes] as f64;
                delta[mes] = soma_delta[mes] / n[mes] as f64;
            }
        }

        PerfilMensal {
            media,
            delta,
            fallback_media,
            fallback_delta,
        }
    }

    fn media(&self, mes: usize) -> f64 {
        self.media.get(mes).copied().unwrap_or(self.fallback_media)
    }

    fn delta(&self, mes: usize) -> f64 {
        self.delta.get(mes).copied().unwrap_or(self.fallback_delta)
    }
}

#[derive(Clone)]
struct RidgeLinear {
    beta: Vec<f64>,
}

impl RidgeLinear {
    fn ajustar(xs: &[Vec<f64>], ys: &[f64], lambda: f64) -> Self {
        if xs.is_empty() {
            return RidgeLinear { beta: vec![0.0] };
        }

        let p = xs[0].len() + 1; // intercepto
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];

        for (x, y) in xs.iter().zip(ys) {
            let mut linha = Vec::with_capacity(p);
            linha.push(1.0);
            linha.extend(x.iter().copied());

            for i in 0..p {
                xty[i] += linha[i] * y;
                for j in 0..p {
                    xtx[i][j] += linha[i] * linha[j];
                }
            }
        }

        for (i, linha) in xtx.iter_mut().enumerate().skip(1) {
            linha[i] += lambda;
        }

        let beta = resolver_sistema_linear(xtx, xty).unwrap_or_else(|| {
            let media = ys.iter().sum::<f64>() / ys.len() as f64;
            let mut b = vec![0.0; p];
            b[0] = media;
            b
        });
        RidgeLinear { beta }
    }

    fn prever(&self, x: &[f64]) -> f64 {
        self.beta[0]
            + x.iter()
                .zip(self.beta.iter().skip(1))
                .map(|(xi, bi)| xi * bi)
                .sum::<f64>()
    }
}

pub fn explorar_plato_temporal(
    registros_treino: &[Registro],
    registros_teste: &[Registro],
    fis: &Fisica,
) {
    println!("\n===========================================");
    println!("  Exploracao: memoria temporal e residual");
    println!("===========================================");

    let mut registros = registros_treino.to_vec();
    registros.extend_from_slice(registros_teste);
    registros.sort_by(|a, b| a.data.cmp(&b.data));

    let linhas = construir_linhas_temporais(&registros);
    let treino: Vec<LinhaTemporal> = linhas
        .iter()
        .filter(|l| l.data.as_str() <= "2022-12-31")
        .cloned()
        .collect();
    let teste: Vec<LinhaTemporal> = linhas
        .iter()
        .filter(|l| l.data.as_str() >= "2023-01-01")
        .cloned()
        .collect();

    if treino.is_empty() || teste.is_empty() {
        println!("[exploracao] sem dados suficientes para MAX_LAG={MAX_LAG}");
        return;
    }

    println!(
        "[exploracao] amostras com janela historica: treino={} | teste={} | max_lag={} dias",
        treino.len(),
        teste.len(),
        MAX_LAG
    );

    let _ = std::fs::create_dir_all(DIR_OUTPUTS);
    salvar_features_temporais();

    let perfil = PerfilMensal::ajustar(&treino);
    let mut resultados = Vec::new();
    let mut pesos_lineares = Vec::new();

    registrar_resultados(
        &mut resultados,
        "persistencia_dia_anterior".to_string(),
        "baseline",
        "nao_aplica",
        &treino,
        &teste,
        0.0,
        "P_t = P_{t-1}".to_string(),
        |l| l.geracao_lag1,
    );
    registrar_resultados(
        &mut resultados,
        "persistencia_delta_mensal".to_string(),
        "residuo_sobre_persistencia",
        "nao_aplica",
        &treino,
        &teste,
        0.0,
        "P_t = P_{t-1} + media_treino(P_t-P_{t-1}|mes)".to_string(),
        |l| limitar(fis, l.geracao_lag1 + perfil.delta(l.mes)),
    );
    registrar_resultados(
        &mut resultados,
        "media_mensal_historica".to_string(),
        "anomalia_mensal",
        "nao_aplica",
        &treino,
        &teste,
        0.0,
        "media mensal ajustada somente no treino".to_string(),
        |l| perfil.media(l.mes),
    );

    let mut predicoes_hidro_longo_salvas = false;

    for (idx_grupo, grupo) in GrupoFeatures::todos().iter().copied().enumerate() {
        let escalador = Escalador::ajustar(&treino, grupo);
        let x_treino = matriz_normalizada(&treino, grupo, &escalador);
        let n_features = grupo.nomes_features().len();
        let observacao_features = if grupo == GrupoFeatures::Operacional {
            "usa apenas t-1 ou anterior, incluindo geracao passada"
        } else {
            "usa apenas hidrologia defasada e calendario"
        };

        println!(
            "[exploracao] grupo={} | features={} | {}",
            grupo.nome(),
            n_features,
            observacao_features
        );

        let y_abs: Vec<f64> = treino.iter().map(|l| alvo_absoluto(l, fis)).collect();
        let modelos_abs = treinar_familia_modelos(
            "absoluto",
            grupo,
            &escalador,
            &treino,
            &teste,
            &x_treino,
            &y_abs,
            fis,
            &perfil,
            100 + idx_grupo as u64,
            &mut resultados,
            &mut pesos_lineares,
        );

        let y_anomalia: Vec<f64> = treino
            .iter()
            .map(|l| alvo_anomalia_mensal(l, fis, &perfil))
            .collect();
        let modelos_anomalia = treinar_familia_modelos(
            "anomalia_mensal",
            grupo,
            &escalador,
            &treino,
            &teste,
            &x_treino,
            &y_anomalia,
            fis,
            &perfil,
            200 + idx_grupo as u64,
            &mut resultados,
            &mut pesos_lineares,
        );

        let y_residual: Vec<f64> = treino
            .iter()
            .map(|l| alvo_residuo_persistencia(l, fis))
            .collect();
        let modelos_residual = treinar_familia_modelos(
            "residuo_sobre_persistencia",
            grupo,
            &escalador,
            &treino,
            &teste,
            &x_treino,
            &y_residual,
            fis,
            &perfil,
            300 + idx_grupo as u64,
            &mut resultados,
            &mut pesos_lineares,
        );

        if grupo == GrupoFeatures::HidroLongo && !predicoes_hidro_longo_salvas {
            salvar_predicoes_hidro_longo(
                &teste,
                fis,
                &perfil,
                &escalador,
                &modelos_abs,
                &modelos_anomalia,
                &modelos_residual,
            );
            predicoes_hidro_longo_salvas = true;
        }
    }

    let fatores = explorar_fator_efetivo(&treino, &teste, fis, &mut resultados);
    let resultados_moveis = validacao_movel_operacional(&linhas, fis);

    salvar_resultados_exploracao(&resultados);
    salvar_pesos_lineares(&pesos_lineares);
    salvar_fatores_efetivos(&fatores);
    salvar_validacao_movel_operacional(&resultados_moveis);
    imprimir_top_resultados(&resultados);
    imprimir_resumo_movel_operacional(&resultados_moveis);
    println!(
        "[exploracao] artefatos: exploracao_temporal_residual.csv, validacao_movel_operacional.csv, exploracao_fator_efetivo.csv, exploracao_features_temporais.csv"
    );
}

#[derive(Clone)]
struct ModelosFamilia {
    ridge: RidgeLinear,
    rf: RandomForestRegressor,
    mlp: Rede,
    formulacao: &'static str,
    grupo: GrupoFeatures,
}

fn treinar_familia_modelos(
    formulacao: &'static str,
    grupo: GrupoFeatures,
    escalador: &Escalador,
    treino: &[LinhaTemporal],
    teste: &[LinhaTemporal],
    x_treino: &[Vec<f64>],
    y_treino: &[f64],
    fis: &Fisica,
    perfil: &PerfilMensal,
    seed: u64,
    resultados: &mut Vec<ResultadoExperimento>,
    pesos_lineares: &mut Vec<PesoLinear>,
) -> ModelosFamilia {
    let nome_grupo = grupo.nome();
    let ridge = RidgeLinear::ajustar(x_treino, y_treino, 1e-2);
    registrar_pesos_lineares(
        pesos_lineares,
        format!("ridge_{formulacao}"),
        formulacao,
        grupo,
        &ridge,
    );
    registrar_resultados(
        resultados,
        format!("ridge_{formulacao}"),
        formulacao,
        nome_grupo,
        treino,
        teste,
        0.0,
        "regressao linear ridge sobre features normalizadas".to_string(),
        |l| {
            let x = escalador.transformar(grupo.entrada(l));
            decodificar_previsao(formulacao, ridge.prever(&x), l, fis, perfil)
        },
    );

    let inicio = Instant::now();
    let rf = RandomForestRegressor::treinar(
        &amostras_arvore(x_treino, y_treino),
        params_rf(grupo.nomes_features().len(), seed + 17),
    );
    let tempo_rf = inicio.elapsed().as_secs_f64();
    registrar_resultados(
        resultados,
        format!("rf_{formulacao}"),
        formulacao,
        nome_grupo,
        treino,
        teste,
        tempo_rf,
        "Random Forest implementada em Rust".to_string(),
        |l| {
            let x = escalador.transformar(grupo.entrada(l));
            decodificar_previsao(formulacao, rf.prever(&x), l, fis, perfil)
        },
    );

    let inicio = Instant::now();
    let mlp = treinar_mlp(x_treino, y_treino, seed + 31);
    let tempo_mlp = inicio.elapsed().as_secs_f64();
    registrar_resultados(
        resultados,
        format!("mlp_{formulacao}"),
        formulacao,
        nome_grupo,
        treino,
        teste,
        tempo_mlp,
        format!(
            "MLP janela temporal simples; epocas={EPOCAS_MLP_EXPLORACAO}; sem variaveis futuras"
        ),
        |l| {
            let x = escalador.transformar(grupo.entrada(l));
            decodificar_previsao(formulacao, prever(&mlp, &x)[0], l, fis, perfil)
        },
    );

    ModelosFamilia {
        ridge,
        rf,
        mlp,
        formulacao,
        grupo,
    }
}

fn construir_linhas_temporais(registros: &[Registro]) -> Vec<LinhaTemporal> {
    let mut linhas = Vec::new();
    if registros.len() <= MAX_LAG {
        return linhas;
    }

    for i in MAX_LAG..registros.len() {
        let r = &registros[i];
        let mes = mes_data(&r.data);

        let q_lag1 = lag(registros, i, 1, |r| r.vazao);
        let q_lag2 = lag(registros, i, 2, |r| r.vazao);
        let q_lag3 = lag(registros, i, 3, |r| r.vazao);
        let q_lag7 = lag(registros, i, 7, |r| r.vazao);
        let q_lag14 = lag(registros, i, 14, |r| r.vazao);
        let q_lag30 = lag(registros, i, 30, |r| r.vazao);
        let q_ma3 = media_lag(registros, i, 3, |r| r.vazao);
        let q_ma7 = media_lag(registros, i, 7, |r| r.vazao);
        let q_ma14 = media_lag(registros, i, 14, |r| r.vazao);
        let q_ma30 = media_lag(registros, i, 30, |r| r.vazao);

        let ena_lag1 = lag(registros, i, 1, |r| r.ena_bruta);
        let ena_lag2 = lag(registros, i, 2, |r| r.ena_bruta);
        let ena_lag3 = lag(registros, i, 3, |r| r.ena_bruta);
        let ena_lag7 = lag(registros, i, 7, |r| r.ena_bruta);
        let ena_lag14 = lag(registros, i, 14, |r| r.ena_bruta);
        let ena_lag30 = lag(registros, i, 30, |r| r.ena_bruta);
        let ena_ma3 = media_lag(registros, i, 3, |r| r.ena_bruta);
        let ena_ma7 = media_lag(registros, i, 7, |r| r.ena_bruta);
        let ena_ma14 = media_lag(registros, i, 14, |r| r.ena_bruta);
        let ena_ma30 = media_lag(registros, i, 30, |r| r.ena_bruta);

        let p_lag1 = lag(registros, i, 1, |r| r.geracao);
        let p_lag2 = lag(registros, i, 2, |r| r.geracao);
        let p_lag7 = lag(registros, i, 7, |r| r.geracao);
        let p_ma3 = media_lag(registros, i, 3, |r| r.geracao);
        let p_ma7 = media_lag(registros, i, 7, |r| r.geracao);

        let hidro_basico = vec![r.mes_sin, r.mes_cos, q_lag1, ena_lag1];
        let mut hidro_longo = vec![
            r.mes_sin,
            r.mes_cos,
            q_lag1,
            q_lag2,
            q_lag3,
            q_lag7,
            q_lag14,
            q_lag30,
            q_ma3,
            q_ma7,
            q_ma14,
            q_ma30,
            q_lag1 - q_lag2,
            q_lag1 - q_lag7,
            ena_lag1,
            ena_lag2,
            ena_lag3,
            ena_lag7,
            ena_lag14,
            ena_lag30,
            ena_ma3,
            ena_ma7,
            ena_ma14,
            ena_ma30,
            ena_lag1 - ena_lag2,
            ena_lag1 - ena_lag7,
        ];
        let mut operacional = hidro_longo.clone();
        operacional.extend([
            p_lag1,
            p_lag2,
            p_lag7,
            p_ma3,
            p_ma7,
            p_lag1 - p_lag2,
            p_lag1 - p_lag7,
        ]);

        debug_assert_eq!(hidro_basico.len(), FEATURES_HIDRO_BASICO.len());
        debug_assert_eq!(hidro_longo.len(), FEATURES_HIDRO_LONGO.len());
        debug_assert_eq!(operacional.len(), FEATURES_OPERACIONAL.len());

        linhas.push(LinhaTemporal {
            data: r.data.clone(),
            mes,
            geracao: r.geracao,
            geracao_lag1: p_lag1,
            q_lag1,
            q_ma7,
            q_ma30,
            hidro_basico,
            hidro_longo: {
                hidro_longo.shrink_to_fit();
                hidro_longo
            },
            operacional,
        });
    }

    linhas
}

fn matriz_normalizada(
    linhas: &[LinhaTemporal],
    grupo: GrupoFeatures,
    escalador: &Escalador,
) -> Vec<Vec<f64>> {
    linhas
        .iter()
        .map(|l| escalador.transformar(grupo.entrada(l)))
        .collect()
}

fn alvo_absoluto(linha: &LinhaTemporal, fis: &Fisica) -> f64 {
    (linha.geracao - fis.p_min) / (fis.p_max - fis.p_min)
}

fn alvo_anomalia_mensal(linha: &LinhaTemporal, fis: &Fisica, perfil: &PerfilMensal) -> f64 {
    (linha.geracao - perfil.media(linha.mes)) / (fis.p_max - fis.p_min)
}

fn alvo_residuo_persistencia(linha: &LinhaTemporal, fis: &Fisica) -> f64 {
    (linha.geracao - linha.geracao_lag1) / (fis.p_max - fis.p_min)
}

fn decodificar_previsao(
    formulacao: &str,
    y_norm: f64,
    linha: &LinhaTemporal,
    fis: &Fisica,
    perfil: &PerfilMensal,
) -> f64 {
    let dp = fis.p_max - fis.p_min;
    let p = match formulacao {
        "absoluto" => fis.p_min + y_norm * dp,
        "anomalia_mensal" => perfil.media(linha.mes) + y_norm * dp,
        "residuo_sobre_persistencia" => linha.geracao_lag1 + y_norm * dp,
        _ => fis.p_min + y_norm * dp,
    };
    limitar(fis, p)
}

fn treinar_mlp(xs: &[Vec<f64>], ys: &[f64], seed: u64) -> Rede {
    let entrada = xs.first().map(|x| x.len()).unwrap_or(1);
    let arquitetura = if entrada <= 8 {
        vec![entrada, 16, 16, 1]
    } else {
        vec![entrada, 24, 16, 1]
    };
    let mut rede = Rede::nova(&arquitetura, seed);
    let lote: Vec<(Vec<f64>, Vec<f64>)> = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x.clone(), vec![*y]))
        .collect();
    let mut adam = Adam::novo(&rede, LR_MLP_EXPLORACAO);

    for epoca in 0..=EPOCAS_MLP_EXPLORACAO {
        let (grads, _) = gradientes_lote(&rede, &lote);
        if epoca < EPOCAS_MLP_EXPLORACAO {
            adam.passo(&mut rede, &grads);
        }
    }

    rede
}

fn amostras_arvore(xs: &[Vec<f64>], ys: &[f64]) -> Vec<AmostraArvore> {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| AmostraArvore {
            x: x.clone(),
            y: *y,
        })
        .collect()
}

fn params_rf(n_features: usize, seed: u64) -> ParametrosFloresta {
    ParametrosFloresta {
        n_arvores: 80,
        max_depth: 9,
        min_samples_split: 30,
        min_samples_leaf: 12,
        mtry: (n_features as f64).sqrt().ceil().max(2.0) as usize,
        seed,
    }
}

fn registrar_resultados<F>(
    resultados: &mut Vec<ResultadoExperimento>,
    modelo: String,
    formulacao: &'static str,
    grupo_features: &str,
    treino: &[LinhaTemporal],
    teste: &[LinhaTemporal],
    tempo_treino_s: f64,
    observacao: String,
    predizer: F,
) where
    F: Fn(&LinhaTemporal) -> f64,
{
    for (periodo, linhas) in [("treino_2015_2022", treino), ("teste_2023_2024", teste)] {
        let (rmse, mae, r2, bias) = calcular_metricas(linhas, &predizer);
        resultados.push(ResultadoExperimento {
            modelo: modelo.clone(),
            formulacao,
            grupo_features: grupo_features.to_string(),
            periodo,
            n: linhas.len(),
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            bias_mwmed: bias,
            tempo_treino_s,
            observacao: observacao.clone(),
        });
    }
}

fn calcular_metricas<F>(linhas: &[LinhaTemporal], predizer: F) -> (f64, f64, f64, f64)
where
    F: Fn(&LinhaTemporal) -> f64,
{
    let n = linhas.len();
    let media_real = linhas.iter().map(|l| l.geracao).sum::<f64>() / n as f64;
    let mut sq = 0.0;
    let mut abs = 0.0;
    let mut bias = 0.0;
    let mut sst = 0.0;

    for linha in linhas {
        let pred = predizer(linha);
        let erro = pred - linha.geracao;
        sq += erro * erro;
        abs += erro.abs();
        bias += erro;
        sst += (linha.geracao - media_real).powi(2);
    }

    let r2 = if sst > 0.0 { 1.0 - sq / sst } else { f64::NAN };
    ((sq / n as f64).sqrt(), abs / n as f64, r2, bias / n as f64)
}

fn registrar_pesos_lineares(
    pesos: &mut Vec<PesoLinear>,
    modelo: String,
    formulacao: &'static str,
    grupo: GrupoFeatures,
    ridge: &RidgeLinear,
) {
    pesos.push(PesoLinear {
        modelo: modelo.clone(),
        formulacao,
        grupo_features: grupo.nome().to_string(),
        feature: "intercepto".to_string(),
        peso: ridge.beta[0],
    });
    for (nome, peso) in grupo.nomes_features().iter().zip(ridge.beta.iter().skip(1)) {
        pesos.push(PesoLinear {
            modelo: modelo.clone(),
            formulacao,
            grupo_features: grupo.nome().to_string(),
            feature: (*nome).to_string(),
            peso: *peso,
        });
    }
}

fn explorar_fator_efetivo(
    treino: &[LinhaTemporal],
    teste: &[LinhaTemporal],
    fis: &Fisica,
    resultados: &mut Vec<ResultadoExperimento>,
) -> Vec<FatorEfetivo> {
    let mut fatores = Vec::new();

    for (variavel, extrair) in [
        ("q_lag1", extrair_q_lag1 as fn(&LinhaTemporal) -> f64),
        ("q_ma7", extrair_q_ma7 as fn(&LinhaTemporal) -> f64),
        ("q_ma30", extrair_q_ma30 as fn(&LinhaTemporal) -> f64),
    ] {
        let (a, b) = ajustar_linear_1d(treino, extrair);
        registrar_resultados(
            resultados,
            format!("fator_efetivo_{variavel}"),
            "fator_efetivo",
            variavel,
            treino,
            teste,
            0.0,
            "P ~= a + k_efetivo Q, ajustado no treino com variavel defasada".to_string(),
            |l| limitar(fis, a + b * extrair(l)),
        );
        let (rmse, mae, r2, _) = calcular_metricas(teste, |l| limitar(fis, a + b * extrair(l)));
        let k_fisico = k_fisico_mw_por_m3s();
        fatores.push(FatorEfetivo {
            variavel,
            intercepto: a,
            k_efetivo_mw_por_m3s: b,
            k_fisico_mw_por_m3s: k_fisico,
            razao_k_efetivo_k_fisico: b / k_fisico,
            rmse_teste_mwmed: rmse,
            mae_teste_mwmed: mae,
            r2_teste: r2,
        });
    }

    fatores
}

fn validacao_movel_operacional(
    linhas: &[LinhaTemporal],
    fis: &Fisica,
) -> Vec<ResultadoMovelOperacional> {
    let grupo = GrupoFeatures::Operacional;
    let mut resultados = Vec::new();

    println!("[exploracao:movel] validacao anual do cenário operacional autorregressivo");

    for ano_teste in 2020..=2024 {
        let treino: Vec<LinhaTemporal> = linhas
            .iter()
            .filter(|l| ano_data(&l.data) < ano_teste)
            .cloned()
            .collect();
        let teste: Vec<LinhaTemporal> = linhas
            .iter()
            .filter(|l| ano_data(&l.data) == ano_teste)
            .cloned()
            .collect();

        if treino.is_empty() || teste.is_empty() {
            continue;
        }

        let perfil = PerfilMensal::ajustar(&treino);
        let escalador = Escalador::ajustar(&treino, grupo);
        let x_treino = matriz_normalizada(&treino, grupo, &escalador);
        let y_residual: Vec<f64> = treino
            .iter()
            .map(|l| alvo_residuo_persistencia(l, fis))
            .collect();
        let y_abs: Vec<f64> = treino.iter().map(|l| alvo_absoluto(l, fis)).collect();

        let persistencia_rmse = registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "persistencia_dia_anterior".to_string(),
            "baseline",
            &treino,
            &teste,
            0.0,
            0.0,
            "P_t = P_{t-1}".to_string(),
            |l| l.geracao_lag1,
        );

        registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "persistencia_delta_mensal".to_string(),
            "residuo_sobre_persistencia",
            &treino,
            &teste,
            persistencia_rmse,
            0.0,
            "P_t = P_{t-1} + media_treino(P_t-P_{t-1}|mes)".to_string(),
            |l| limitar(fis, l.geracao_lag1 + perfil.delta(l.mes)),
        );

        let ridge_residual = RidgeLinear::ajustar(&x_treino, &y_residual, 1e-2);
        registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "ridge_residuo_sobre_persistencia".to_string(),
            "residuo_sobre_persistencia",
            &treino,
            &teste,
            persistencia_rmse,
            0.0,
            "ridge operacional com geracao passada e hidrologia defasada".to_string(),
            |l| {
                let x = escalador.transformar(grupo.entrada(l));
                decodificar_previsao(
                    "residuo_sobre_persistencia",
                    ridge_residual.prever(&x),
                    l,
                    fis,
                    &perfil,
                )
            },
        );

        let inicio = Instant::now();
        let rf_residual = RandomForestRegressor::treinar(
            &amostras_arvore(&x_treino, &y_residual),
            params_rf(grupo.nomes_features().len(), 700 + ano_teste as u64),
        );
        let tempo_rf = inicio.elapsed().as_secs_f64();
        registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "rf_residuo_sobre_persistencia".to_string(),
            "residuo_sobre_persistencia",
            &treino,
            &teste,
            persistencia_rmse,
            tempo_rf,
            "Random Forest residual operacional".to_string(),
            |l| {
                let x = escalador.transformar(grupo.entrada(l));
                decodificar_previsao(
                    "residuo_sobre_persistencia",
                    rf_residual.prever(&x),
                    l,
                    fis,
                    &perfil,
                )
            },
        );

        let inicio = Instant::now();
        let mlp_residual = treinar_mlp(&x_treino, &y_residual, 800 + ano_teste as u64);
        let tempo_mlp_residual = inicio.elapsed().as_secs_f64();
        let rmse_mlp_residual = registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "mlp_residuo_sobre_persistencia".to_string(),
            "residuo_sobre_persistencia",
            &treino,
            &teste,
            persistencia_rmse,
            tempo_mlp_residual,
            format!(
                "MLP residual operacional; epocas={EPOCAS_MLP_EXPLORACAO}; sem variaveis futuras"
            ),
            |l| {
                let x = escalador.transformar(grupo.entrada(l));
                decodificar_previsao(
                    "residuo_sobre_persistencia",
                    prever(&mlp_residual, &x)[0],
                    l,
                    fis,
                    &perfil,
                )
            },
        );

        let ridge_abs = RidgeLinear::ajustar(&x_treino, &y_abs, 1e-2);
        registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "ridge_absoluto".to_string(),
            "absoluto",
            &treino,
            &teste,
            persistencia_rmse,
            0.0,
            "ridge operacional prevendo P_t diretamente".to_string(),
            |l| {
                let x = escalador.transformar(grupo.entrada(l));
                decodificar_previsao("absoluto", ridge_abs.prever(&x), l, fis, &perfil)
            },
        );

        let inicio = Instant::now();
        let mlp_abs = treinar_mlp(&x_treino, &y_abs, 900 + ano_teste as u64);
        let tempo_mlp_abs = inicio.elapsed().as_secs_f64();
        registrar_movel_operacional(
            &mut resultados,
            ano_teste,
            "mlp_absoluto".to_string(),
            "absoluto",
            &treino,
            &teste,
            persistencia_rmse,
            tempo_mlp_abs,
            format!(
                "MLP operacional prevendo P_t; epocas={EPOCAS_MLP_EXPLORACAO}; sem variaveis futuras"
            ),
            |l| {
                let x = escalador.transformar(grupo.entrada(l));
                decodificar_previsao("absoluto", prever(&mlp_abs, &x)[0], l, fis, &perfil)
            },
        );

        println!(
            "[exploracao:movel] ano {ano_teste}: persistencia={persistencia_rmse:.0} | mlp_residual={rmse_mlp_residual:.0} MWmed"
        );
    }

    resultados
}

fn registrar_movel_operacional<F>(
    resultados: &mut Vec<ResultadoMovelOperacional>,
    ano_teste: i32,
    modelo: String,
    formulacao: &'static str,
    treino: &[LinhaTemporal],
    teste: &[LinhaTemporal],
    persistencia_rmse: f64,
    tempo_treino_s: f64,
    observacao: String,
    predizer: F,
) -> f64
where
    F: Fn(&LinhaTemporal) -> f64,
{
    let (rmse, mae, r2, bias) = calcular_metricas(teste, predizer);
    let melhoria = if persistencia_rmse > 0.0 {
        100.0 * (persistencia_rmse - rmse) / persistencia_rmse
    } else {
        0.0
    };
    resultados.push(ResultadoMovelOperacional {
        ano_teste,
        modelo,
        formulacao,
        n_treino: treino.len(),
        n_teste: teste.len(),
        rmse_mwmed: rmse,
        mae_mwmed: mae,
        r2,
        bias_mwmed: bias,
        melhoria_rmse_vs_persistencia_pct: melhoria,
        tempo_treino_s,
        observacao,
    });
    rmse
}

fn ajustar_linear_1d(linhas: &[LinhaTemporal], extrair: fn(&LinhaTemporal) -> f64) -> (f64, f64) {
    let n = linhas.len() as f64;
    let mx = linhas.iter().map(extrair).sum::<f64>() / n;
    let my = linhas.iter().map(|l| l.geracao).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for linha in linhas {
        let dx = extrair(linha) - mx;
        cov += dx * (linha.geracao - my);
        var += dx * dx;
    }
    let b = if var > 1e-12 { cov / var } else { 0.0 };
    let a = my - b * mx;
    (a, b)
}

fn resolver_sistema_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for lin in (col + 1)..n {
            if a[lin][col].abs() > a[pivot][col].abs() {
                pivot = lin;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let div = a[col][col];
        for j in col..n {
            a[col][j] /= div;
        }
        b[col] /= div;

        for lin in 0..n {
            if lin == col {
                continue;
            }
            let fator = a[lin][col];
            if fator.abs() < 1e-18 {
                continue;
            }
            for j in col..n {
                a[lin][j] -= fator * a[col][j];
            }
            b[lin] -= fator * b[col];
        }
    }
    Some(b)
}

fn salvar_predicoes_hidro_longo(
    teste: &[LinhaTemporal],
    fis: &Fisica,
    perfil: &PerfilMensal,
    escalador: &Escalador,
    modelos_abs: &ModelosFamilia,
    modelos_anomalia: &ModelosFamilia,
    modelos_residual: &ModelosFamilia,
) {
    let caminho = Path::new(DIR_OUTPUTS).join("predicoes_exploracao_temporal.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "data",
            "geracao_real_mwmed",
            "persistencia_mwmed",
            "mlp_absoluto_hidro_longo_mwmed",
            "mlp_anomalia_mensal_hidro_longo_mwmed",
            "ridge_residual_hidro_longo_mwmed",
            "rf_residual_hidro_longo_mwmed",
            "mlp_residual_hidro_longo_mwmed",
        ]);

        for linha in teste {
            let x_abs = escalador.transformar(modelos_abs.grupo.entrada(linha));
            let x_anom = escalador.transformar(modelos_anomalia.grupo.entrada(linha));
            let x_res = escalador.transformar(modelos_residual.grupo.entrada(linha));
            let mlp_abs = decodificar_previsao(
                modelos_abs.formulacao,
                prever(&modelos_abs.mlp, &x_abs)[0],
                linha,
                fis,
                perfil,
            );
            let mlp_anom = decodificar_previsao(
                modelos_anomalia.formulacao,
                prever(&modelos_anomalia.mlp, &x_anom)[0],
                linha,
                fis,
                perfil,
            );
            let ridge_res = decodificar_previsao(
                modelos_residual.formulacao,
                modelos_residual.ridge.prever(&x_res),
                linha,
                fis,
                perfil,
            );
            let rf_res = decodificar_previsao(
                modelos_residual.formulacao,
                modelos_residual.rf.prever(&x_res),
                linha,
                fis,
                perfil,
            );
            let mlp_res = decodificar_previsao(
                modelos_residual.formulacao,
                prever(&modelos_residual.mlp, &x_res)[0],
                linha,
                fis,
                perfil,
            );
            let _ = w.write_record(&[
                linha.data.clone(),
                format!("{:.4}", linha.geracao),
                format!("{:.4}", linha.geracao_lag1),
                format!("{:.4}", mlp_abs),
                format!("{:.4}", mlp_anom),
                format!("{:.4}", ridge_res),
                format!("{:.4}", rf_res),
                format!("{:.4}", mlp_res),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_features_temporais() {
    let caminho = Path::new(DIR_OUTPUTS).join("exploracao_features_temporais.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record(["grupo_features", "ordem", "feature", "disponibilidade"]);
        for grupo in GrupoFeatures::todos() {
            for (i, feature) in grupo.nomes_features().iter().enumerate() {
                let disponibilidade = if feature.starts_with("mes") {
                    "calendario conhecido antes do dia previsto"
                } else {
                    "observacao em t-1 ou anterior"
                };
                let _ = w.write_record(&[
                    grupo.nome().to_string(),
                    (i + 1).to_string(),
                    (*feature).to_string(),
                    disponibilidade.to_string(),
                ]);
            }
        }
        let _ = w.flush();
    }
}

fn salvar_resultados_exploracao(resultados: &[ResultadoExperimento]) {
    let caminho = Path::new(DIR_OUTPUTS).join("exploracao_temporal_residual.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "modelo",
            "formulacao",
            "grupo_features",
            "periodo",
            "n",
            "rmse_mwmed",
            "mae_mwmed",
            "r2",
            "bias_mwmed",
            "tempo_treino_s",
            "observacao",
        ]);
        for r in resultados {
            let _ = w.write_record(&[
                r.modelo.clone(),
                r.formulacao.to_string(),
                r.grupo_features.clone(),
                r.periodo.to_string(),
                r.n.to_string(),
                format!("{:.2}", r.rmse_mwmed),
                format!("{:.2}", r.mae_mwmed),
                format!("{:.4}", r.r2),
                format!("{:.2}", r.bias_mwmed),
                format!("{:.6}", r.tempo_treino_s),
                r.observacao.clone(),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_pesos_lineares(pesos: &[PesoLinear]) {
    let caminho = Path::new(DIR_OUTPUTS).join("exploracao_pesos_lineares.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record(["modelo", "formulacao", "grupo_features", "feature", "peso"]);
        for p in pesos {
            let _ = w.write_record(&[
                p.modelo.clone(),
                p.formulacao.to_string(),
                p.grupo_features.clone(),
                p.feature.clone(),
                format!("{:.10}", p.peso),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_fatores_efetivos(fatores: &[FatorEfetivo]) {
    let caminho = Path::new(DIR_OUTPUTS).join("exploracao_fator_efetivo.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "variavel",
            "intercepto",
            "k_efetivo_mw_por_m3s",
            "k_fisico_mw_por_m3s",
            "razao_k_efetivo_k_fisico",
            "rmse_teste_mwmed",
            "mae_teste_mwmed",
            "r2_teste",
        ]);
        for f in fatores {
            let _ = w.write_record(&[
                f.variavel.to_string(),
                format!("{:.6}", f.intercepto),
                format!("{:.8}", f.k_efetivo_mw_por_m3s),
                format!("{:.8}", f.k_fisico_mw_por_m3s),
                format!("{:.6}", f.razao_k_efetivo_k_fisico),
                format!("{:.2}", f.rmse_teste_mwmed),
                format!("{:.2}", f.mae_teste_mwmed),
                format!("{:.4}", f.r2_teste),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_validacao_movel_operacional(resultados: &[ResultadoMovelOperacional]) {
    let caminho = Path::new(DIR_OUTPUTS).join("validacao_movel_operacional.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "ano_teste",
            "modelo",
            "formulacao",
            "n_treino",
            "n_teste",
            "rmse_mwmed",
            "mae_mwmed",
            "r2",
            "bias_mwmed",
            "melhoria_rmse_vs_persistencia_pct",
            "tempo_treino_s",
            "observacao",
        ]);
        for r in resultados {
            let _ = w.write_record(&[
                r.ano_teste.to_string(),
                r.modelo.clone(),
                r.formulacao.to_string(),
                r.n_treino.to_string(),
                r.n_teste.to_string(),
                format!("{:.2}", r.rmse_mwmed),
                format!("{:.2}", r.mae_mwmed),
                format!("{:.4}", r.r2),
                format!("{:.2}", r.bias_mwmed),
                format!("{:.2}", r.melhoria_rmse_vs_persistencia_pct),
                format!("{:.6}", r.tempo_treino_s),
                r.observacao.clone(),
            ]);
        }
        let _ = w.flush();
    }
}

fn imprimir_top_resultados(resultados: &[ResultadoExperimento]) {
    let mut teste: Vec<&ResultadoExperimento> = resultados
        .iter()
        .filter(|r| r.periodo == "teste_2023_2024")
        .collect();
    teste.sort_by(|a, b| a.rmse_mwmed.partial_cmp(&b.rmse_mwmed).unwrap());

    println!("[exploracao] melhores modelos no teste 2023-2024:");
    for r in teste.iter().take(10) {
        println!(
            "[exploracao] {:<28} | {:<26} | RMSE={:>7.0} | MAE={:>7.0} | R2={:>6.3}",
            r.modelo, r.grupo_features, r.rmse_mwmed, r.mae_mwmed, r.r2
        );
    }
}

fn imprimir_resumo_movel_operacional(resultados: &[ResultadoMovelOperacional]) {
    let modelos = [
        "persistencia_dia_anterior",
        "ridge_residuo_sobre_persistencia",
        "rf_residuo_sobre_persistencia",
        "mlp_residuo_sobre_persistencia",
        "mlp_absoluto",
    ];

    println!("[exploracao:movel] medias 2020-2024:");
    for modelo in modelos {
        let linhas: Vec<&ResultadoMovelOperacional> =
            resultados.iter().filter(|r| r.modelo == modelo).collect();
        if linhas.is_empty() {
            continue;
        }
        let n = linhas.len() as f64;
        let rmse = linhas.iter().map(|r| r.rmse_mwmed).sum::<f64>() / n;
        let mae = linhas.iter().map(|r| r.mae_mwmed).sum::<f64>() / n;
        let melhoria = linhas
            .iter()
            .map(|r| r.melhoria_rmse_vs_persistencia_pct)
            .sum::<f64>()
            / n;
        println!(
            "[exploracao:movel] {modelo:<34} RMSE médio={rmse:>7.0} | MAE médio={mae:>7.0} | melhora={melhoria:>5.1}%"
        );
    }
}

fn mes_data(data: &str) -> usize {
    data.get(5..7)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|m| (1..=12).contains(m))
        .unwrap_or(1)
}

fn ano_data(data: &str) -> i32 {
    data.get(0..4)
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0)
}

fn lag<F>(registros: &[Registro], i: usize, atraso: usize, extrair: F) -> f64
where
    F: Fn(&Registro) -> f64,
{
    extrair(&registros[i - atraso])
}

fn media_lag<F>(registros: &[Registro], i: usize, janela: usize, extrair: F) -> f64
where
    F: Fn(&Registro) -> f64,
{
    let ini = i - janela;
    registros[ini..i].iter().map(extrair).sum::<f64>() / janela as f64
}

fn limitar(fis: &Fisica, p: f64) -> f64 {
    p.min(fis.p_max).max(0.0)
}

fn extrair_q_lag1(linha: &LinhaTemporal) -> f64 {
    linha.q_lag1
}

fn extrair_q_ma7(linha: &LinhaTemporal) -> f64 {
    linha.q_ma7
}

fn extrair_q_ma30(linha: &LinhaTemporal) -> f64 {
    linha.q_ma30
}
