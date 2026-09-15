// main.rs — Entry point do projeto PGNN-Itaipu

// Laços com índice explícito espelham a notação matemática do artigo, e as
// funções de experimento recebem a configuração completa como argumentos.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod data;
mod experimentos;
mod math;
mod ml;
mod neural;
mod pinn;
mod simulacao;

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use math::interpolacao;
use math::newton_raphson::{self, Config};
use ml::arvores::{AmostraArvore, ParametrosFloresta, RandomForestRegressor};
use neural::backward::gradientes_lote;
use neural::forward::prever;
use neural::loss::{Fisica, Lambdas, Termos};
use neural::network::Rede;
use neural::optimizer::Adam;

// Diretório de saída resolvido a partir do Cargo.toml (rust/), independente do cwd.
const DIR_OUTPUTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/outputs");

const CAMINHO_DADOS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/processed/dataset_final.csv"
);
const CAMINHO_NORM: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/processed/normalizacao.json"
);
const LIMITE_TREINO_INTERNO: &str = "2020-12-31";
const INICIO_VALIDACAO: &str = "2021-01-01";
const LIMITE_VALIDACAO: &str = "2022-12-31";
const LIMITE_TREINO: &str = "2022-12-31";
const INICIO_TESTE: &str = "2023-01-01";
const EPOCAS_JANELAS_MOVEIS: usize = 600;
const H_FD_DIAGNOSTICO: f64 = 1e-3;

fn main() {
    println!("===========================================");
    println!("  PGNN-Itaipu — Fase 3: matemática base");
    println!("===========================================\n");

    // 1) Carregar dados
    let registros = match data::loader::carregar_csv(CAMINHO_DADOS) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERRO ao carregar {CAMINHO_DADOS}: {e}");
            eprintln!("Rode antes a Fase 2 (scripts Python) para gerar o CSV.");
            std::process::exit(1);
        }
    };
    println!("[dados] {} registros carregados", registros.len());
    let (registros_treino, registros_teste) = dividir_temporal(&registros);
    println!(
        "[dados] split temporal: treino={} (2015–2022) | teste={} (2023–2024)",
        registros_treino.len(),
        registros_teste.len()
    );
    if let Some(r) = registros.first() {
        println!(
            "[dados] primeiro: {} | vazao={:.1} m³/s | geracao={:.1} MWmed",
            r.data, r.vazao, r.geracao
        );
    }

    // 2) Interpolação quadrática
    // Três pontos fixos (independentes do dataset) para validação cruzada
    // com o Python (notebook 04). y = x² esperado.
    let pontos = [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
    let xs_teste = [1.5, 2.5, 2.75];
    println!("\n[interp] pontos {pontos:?}  (esperado y = x²)");
    for x in xs_teste {
        let yl = interpolacao::lagrange_quadratica(pontos, x);
        let yn = interpolacao::newton_quadratica(pontos, x);
        println!("[interp] x={x:>4}  lagrange={yl:.10}  newton={yn:.10}");
    }

    // 3) Newton-Raphson
    let cfg = Config::default();
    // raiz de x² - 2  => sqrt(2)
    if let Some(r) = newton_raphson::raiz(|x| x * x - 2.0, |x| 2.0 * x, 1.0, &cfg) {
        println!(
            "\n[newton] raiz de x²-2 = {r:.12}  (sqrt2 = {:.12})",
            2.0_f64.sqrt()
        );
    }

    // 4) Demo aplicada: extremo da parábola por 3 pontos do dataset
    // Pega 3 pontos (vazao, geracao) bem separados e acha o vértice analítico.
    if registros.len() >= 3 {
        let n = registros.len();
        let p = [
            (registros[0].vazao, registros[0].geracao),
            (registros[n / 2].vazao, registros[n / 2].geracao),
            (registros[n - 1].vazao, registros[n - 1].geracao),
        ];
        let coef = interpolacao::coeficientes_quadratica(p);
        println!(
            "\n[demo] parábola por 3 pontos do dataset: a={:.4e} b={:.4e} c={:.4e}",
            coef.0, coef.1, coef.2
        );
        match interpolacao::vertice_quadratica(coef) {
            Some(v) => println!("[demo] vértice (vazão de extremo) ≈ {v:.1} m³/s"),
            None => println!("[demo] sem vértice (parábola degenerada)"),
        }
    }

    println!("\n✓ Fase 3 OK.");

    // 5) Teste de sanidade da rede neural
    fase4_sanidade();

    // 6) PINN completa sobre os dados reais
    let modelo = fase5_pinn(&registros_treino, &registros_teste);

    // 7) análise retrospectiva temporal (real vs. referência)
    if let Some((rede, fis)) = modelo {
        fase6_retrospectiva(&rede, &fis, &registros_treino, &registros_teste);
    }
}

/// Treina a rede (do zero) para ajustar `y = x²` em [-1, 1].
fn fase4_sanidade() {
    println!("\n===========================================");
    println!("  Fase 4: rede neural — sanidade (y = x²)");
    println!("===========================================");

    // Dataset sintético
    let n = 50usize;
    let lote: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            (vec![x], vec![x * x])
        })
        .collect();

    let mut rede = Rede::nova(&[1, 16, 16, 1], 42);
    let mut adam = Adam::novo(&rede, 0.01);

    let epocas = 3000;
    let mut loss_inicial = 0.0;
    for epoca in 0..=epocas {
        let (grads, loss) = gradientes_lote(&rede, &lote);
        if epoca == 0 {
            loss_inicial = loss;
        }
        if epoca % 500 == 0 {
            println!("[fase4] época {epoca:>4}  MSE = {loss:.6}");
        }
        if epoca < epocas {
            adam.passo(&mut rede, &grads);
        }
    }

    // Verificação pontual em alguns x
    let (_, loss_final) = gradientes_lote(&rede, &lote);
    println!(
        "[fase4] loss {:.6} -> {:.6}  (redução {:.1}×)",
        loss_inicial,
        loss_final,
        loss_inicial / loss_final.max(1e-12)
    );
    for x in [-0.8, -0.2, 0.5, 0.9] {
        let y = prever(&rede, &[x])[0];
        println!("[fase4] x={x:+.2}  rede={y:+.4}  real(x²)={:+.4}", x * x);
    }

    if loss_final < 1e-3 {
        println!("\n✓ Fase 4 OK — rede aprende dados sintéticos (pronta p/ Fase 5).");
    } else {
        println!("\n⚠️ Fase 4: loss final alta ({loss_final:.4}) — revisar.");
    }
}

/// Vazão (m³/s) onde a curva atinge a capacidade (joelho da saturação),
/// ou `None` se a curva não chega a ~99% da capacidade.
fn joelho_saturacao(curva: &[pinn::otimizacao::PontoOtimo], p_max: f64) -> Option<f64> {
    curva
        .iter()
        .find(|p| p.geracao_otima >= 0.99 * p_max)
        .map(|p| p.vazao)
}

fn dividir_temporal(
    registros: &[data::loader::Registro],
) -> (Vec<data::loader::Registro>, Vec<data::loader::Registro>) {
    let treino = registros
        .iter()
        .filter(|r| r.data.as_str() <= LIMITE_TREINO)
        .cloned()
        .collect();
    let teste = registros
        .iter()
        .filter(|r| r.data.as_str() >= INICIO_TESTE)
        .cloned()
        .collect();
    (treino, teste)
}

fn dividir_treino_validacao(
    registros: &[data::loader::Registro],
) -> (Vec<data::loader::Registro>, Vec<data::loader::Registro>) {
    let treino_interno = registros
        .iter()
        .filter(|r| r.data.as_str() <= LIMITE_TREINO_INTERNO)
        .cloned()
        .collect();
    let validacao = registros
        .iter()
        .filter(|r| r.data.as_str() >= INICIO_VALIDACAO && r.data.as_str() <= LIMITE_VALIDACAO)
        .cloned()
        .collect();
    (treino_interno, validacao)
}

fn ano_data(data: &str) -> i32 {
    data.get(0..4)
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0)
}

fn mes_data(data: &str) -> usize {
    data.get(5..7)
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|m| (1..=12).contains(m))
        .unwrap_or(1)
}

#[derive(Debug, Clone, Copy)]
struct MediaConstante {
    valor: f64,
}

impl MediaConstante {
    fn ajustar(registros: &[data::loader::Registro]) -> Self {
        let valor = registros.iter().map(|r| r.geracao).sum::<f64>() / registros.len() as f64;
        MediaConstante { valor }
    }

    fn prever(&self) -> f64 {
        self.valor
    }
}

#[derive(Debug, Clone, Copy)]
struct MediaMensal {
    medias: [f64; 13],
    fallback: f64,
}

impl MediaMensal {
    fn ajustar(registros: &[data::loader::Registro]) -> Self {
        let fallback = registros.iter().map(|r| r.geracao).sum::<f64>() / registros.len() as f64;
        let mut somas = [0.0; 13];
        let mut contagens = [0usize; 13];
        for r in registros {
            let m = mes_data(&r.data);
            somas[m] += r.geracao;
            contagens[m] += 1;
        }
        let mut medias = [fallback; 13];
        for m in 1..=12 {
            if contagens[m] > 0 {
                medias[m] = somas[m] / contagens[m] as f64;
            }
        }
        MediaMensal { medias, fallback }
    }

    fn prever(&self, data: &str) -> f64 {
        let m = mes_data(data);
        self.medias.get(m).copied().unwrap_or(self.fallback)
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearPlato {
    a: f64,
    b: f64,
    p_max: f64,
}

impl LinearPlato {
    fn ajustar(registros: &[data::loader::Registro], p_max: f64) -> Self {
        let n = registros.len() as f64;
        let soma_q: f64 = registros.iter().map(|r| r.vazao).sum();
        let soma_p: f64 = registros.iter().map(|r| r.geracao).sum();
        let soma_q2: f64 = registros.iter().map(|r| r.vazao * r.vazao).sum();
        let soma_qp: f64 = registros.iter().map(|r| r.vazao * r.geracao).sum();
        let den = n * soma_q2 - soma_q * soma_q;
        let b = if den.abs() > 1e-12 {
            (n * soma_qp - soma_q * soma_p) / den
        } else {
            0.0
        };
        let a = (soma_p - b * soma_q) / n;
        LinearPlato { a, b, p_max }
    }

    fn prever(&self, vazao: f64) -> f64 {
        (self.a + self.b * vazao).min(self.p_max).max(0.0)
    }

    fn q_sat(&self) -> Option<f64> {
        if self.b > 0.0 {
            Some((self.p_max - self.a) / self.b)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct QuadraticaReferencia {
    pontos: [(f64, f64); 3],
    coef: (f64, f64, f64),
}

impl QuadraticaReferencia {
    fn ajustar(registros: &[data::loader::Registro]) -> Self {
        let mut ordenados = registros.to_vec();
        ordenados.sort_by(|a, b| a.vazao.partial_cmp(&b.vazao).unwrap());
        let n = ordenados.len();
        let idx = [0, n / 2, n - 1];
        let pontos = idx.map(|i| (ordenados[i].vazao, ordenados[i].geracao));
        let coef = interpolacao::coeficientes_quadratica(pontos);
        QuadraticaReferencia { pontos, coef }
    }

    fn prever(&self, vazao: f64) -> f64 {
        interpolacao::avaliar_quadratica(self.coef, vazao)
    }

    fn vertice(&self) -> Option<f64> {
        interpolacao::vertice_quadratica(self.coef)
    }
}

#[derive(Debug, Clone)]
struct MetricasModelo {
    modelo: &'static str,
    periodo: &'static str,
    n: usize,
    mse_norm: f64,
    rmse_mwmed: f64,
    mae_mwmed: f64,
    r2: f64,
    bias_mwmed: f64,
    satura: String,
}

#[derive(Debug, Clone)]
struct BenchmarkModelo {
    modelo: &'static str,
    n_treino: usize,
    n_teste: usize,
    treino_s: f64,
    inferencia_teste_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct NormalizadorJanela {
    q_min: f64,
    q_max: f64,
    ena_min: f64,
    ena_max: f64,
    qlag_min: f64,
    qlag_max: f64,
    p_min: f64,
    p_max: f64,
}

#[derive(Debug, Clone)]
struct MetricaJanela {
    modelo: &'static str,
    ano_teste: i32,
    n_treino: usize,
    n_teste: usize,
    rmse_mwmed: f64,
    mae_mwmed: f64,
    r2: f64,
    tempo_treino_s: f64,
}

#[derive(Debug, Clone)]
struct SensibilidadeLambda {
    lambda_fisica: f64,
    seed: u64,
    rmse_validacao_mwmed: f64,
    loss_fisica_treino: f64,
    joelho_saturacao_m3s: Option<f64>,
    elegivel: bool,
}

#[derive(Debug, Clone)]
struct DiagnosticoFisico {
    modelo: &'static str,
    conjunto: &'static str,
    n: usize,
    cortes_superiores: usize,
    cortes_inferiores: usize,
    pct_cortado: f64,
    corte_medio_mwmed: f64,
    corte_max_mwmed: f64,
    viol_monotonia_pct: f64,
    viol_monotonia_media_mw_por_m3s: f64,
    derivada_mae_mw_por_m3s: f64,
    derivada_rmse_mw_por_m3s: f64,
    derivada_bias_mw_por_m3s: f64,
}

fn faixa<F>(registros: &[data::loader::Registro], extrair: F) -> (f64, f64)
where
    F: Fn(&data::loader::Registro) -> f64,
{
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for r in registros {
        let v = extrair(r);
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

fn norm_janela(valor: f64, min: f64, max: f64) -> f64 {
    let d = max - min;
    if d.abs() < 1e-12 {
        0.0
    } else {
        (valor - min) / d
    }
}

impl NormalizadorJanela {
    fn ajustar(registros: &[data::loader::Registro]) -> Self {
        let (q_min, q_max) = faixa(registros, |r| r.vazao);
        let (ena_min, ena_max) = faixa(registros, |r| r.ena_bruta);
        let (qlag_min, qlag_max) = faixa(registros, |r| r.vazao_lag1);
        let (p_min, p_max) = faixa(registros, |r| r.geracao);
        NormalizadorJanela {
            q_min,
            q_max,
            ena_min,
            ena_max,
            qlag_min,
            qlag_max,
            p_min,
            p_max,
        }
    }

    fn fisica(&self) -> Fisica {
        Fisica::nova(self.q_min, self.q_max, self.p_min, self.p_max)
    }

    fn entrada_multivariada(&self, r: &data::loader::Registro) -> Vec<f64> {
        vec![
            norm_janela(r.vazao, self.q_min, self.q_max),
            norm_janela(r.ena_bruta, self.ena_min, self.ena_max),
            r.mes_sin,
            r.mes_cos,
            norm_janela(r.vazao_lag1, self.qlag_min, self.qlag_max),
        ]
    }

    fn alvo(&self, r: &data::loader::Registro) -> f64 {
        norm_janela(r.geracao, self.p_min, self.p_max)
    }

    fn q_norm(&self, r: &data::loader::Registro) -> f64 {
        norm_janela(r.vazao, self.q_min, self.q_max)
    }
}

fn entrada_vazao(r: &data::loader::Registro) -> Vec<f64> {
    vec![r.vazao_norm]
}

fn entrada_hidrologica(r: &data::loader::Registro) -> Vec<f64> {
    vec![
        r.vazao_norm,
        r.ena_bruta_norm,
        r.mes_sin,
        r.mes_cos,
        r.vazao_lag1_norm,
    ]
}

fn entrada_multivariada(r: &data::loader::Registro) -> Vec<f64> {
    vec![
        r.vazao_norm,
        r.ena_bruta_norm,
        r.mes_sin,
        r.mes_cos,
        r.vazao_lag1_norm,
    ]
}

fn desnormalizar_geracao(fis: &Fisica, p_norm: f64) -> f64 {
    fis.p_min + p_norm * (fis.p_max - fis.p_min)
}

fn limitar_geracao(fis: &Fisica, p: f64) -> f64 {
    p.min(fis.p_max).max(0.0)
}

fn predizer_rede_bruta(rede: &Rede, fis: &Fisica, entrada: &[f64]) -> f64 {
    desnormalizar_geracao(fis, prever(rede, entrada)[0])
}

fn predizer_rede(rede: &Rede, fis: &Fisica, r: &data::loader::Registro, entrada: Vec<f64>) -> f64 {
    let p_norm = prever(rede, &entrada)[0];
    let (_, p) = fis.desnormalizar(r.vazao_norm, p_norm);
    limitar_geracao(fis, p)
}

fn predizer_pinn(rede: &Rede, fis: &Fisica, r: &data::loader::Registro) -> f64 {
    predizer_rede(rede, fis, r, entrada_vazao(r))
}

fn predizer_multivariada(rede: &Rede, fis: &Fisica, r: &data::loader::Registro) -> f64 {
    predizer_rede(rede, fis, r, entrada_multivariada(r))
}

fn predizer_hidrologica(rede: &Rede, fis: &Fisica, r: &data::loader::Registro) -> f64 {
    predizer_rede(rede, fis, r, entrada_hidrologica(r))
}

fn amostras_arvore<F>(registros: &[data::loader::Registro], entrada: F) -> Vec<AmostraArvore>
where
    F: Fn(&data::loader::Registro) -> Vec<f64>,
{
    registros
        .iter()
        .map(|r| AmostraArvore {
            x: entrada(r),
            y: r.geracao_norm,
        })
        .collect()
}

fn amostras_multivariadas_janela(
    registros: &[data::loader::Registro],
    norm: &NormalizadorJanela,
) -> Vec<data::loader::AmostraModelo> {
    registros
        .iter()
        .map(|r| data::loader::AmostraModelo {
            entrada: norm.entrada_multivariada(r),
            alvo: norm.alvo(r),
            q_norm: norm.q_norm(r),
        })
        .collect()
}

fn amostras_arvore_multivariadas_janela(
    registros: &[data::loader::Registro],
    norm: &NormalizadorJanela,
) -> Vec<AmostraArvore> {
    registros
        .iter()
        .map(|r| AmostraArvore {
            x: norm.entrada_multivariada(r),
            y: norm.alvo(r),
        })
        .collect()
}

fn predizer_floresta(
    floresta: &RandomForestRegressor,
    fis: &Fisica,
    r: &data::loader::Registro,
    entrada: Vec<f64>,
) -> f64 {
    let p_norm = floresta.prever(&entrada);
    let (_, p) = fis.desnormalizar(r.vazao_norm, p_norm);
    limitar_geracao(fis, p)
}

fn predizer_rede_janela(
    rede: &Rede,
    fis: &Fisica,
    norm: &NormalizadorJanela,
    r: &data::loader::Registro,
) -> f64 {
    let entrada = norm.entrada_multivariada(r);
    let p_norm = prever(rede, &entrada)[0];
    let (_, p) = fis.desnormalizar(norm.q_norm(r), p_norm);
    limitar_geracao(fis, p)
}

fn predizer_floresta_janela(
    floresta: &RandomForestRegressor,
    fis: &Fisica,
    norm: &NormalizadorJanela,
    r: &data::loader::Registro,
) -> f64 {
    let entrada = norm.entrada_multivariada(r);
    let p_norm = floresta.prever(&entrada);
    let (_, p) = fis.desnormalizar(norm.q_norm(r), p_norm);
    limitar_geracao(fis, p)
}

fn medir_inferencia<F>(registros: &[data::loader::Registro], predizer: F) -> f64
where
    F: Fn(&data::loader::Registro) -> f64,
{
    let ini = Instant::now();
    let mut soma = 0.0;
    for r in registros {
        soma += predizer(r);
    }
    black_box(soma);
    ini.elapsed().as_secs_f64() * 1000.0
}

fn calcular_metricas_mw<F>(registros: &[data::loader::Registro], predizer: F) -> (f64, f64, f64)
where
    F: Fn(&data::loader::Registro) -> f64,
{
    let n = registros.len();
    let media_real = registros.iter().map(|r| r.geracao).sum::<f64>() / n as f64;
    let mut soma_quad = 0.0;
    let mut soma_abs = 0.0;
    let mut sst = 0.0;

    for r in registros {
        let pred = predizer(r);
        let erro = pred - r.geracao;
        soma_quad += erro * erro;
        soma_abs += erro.abs();
        sst += (r.geracao - media_real).powi(2);
    }

    let rmse = (soma_quad / n as f64).sqrt();
    let mae = soma_abs / n as f64;
    let r2 = if sst > 0.0 {
        1.0 - soma_quad / sst
    } else {
        f64::NAN
    };
    (rmse, mae, r2)
}

fn calcular_metricas<F>(
    modelo: &'static str,
    periodo: &'static str,
    registros: &[data::loader::Registro],
    fis: &Fisica,
    satura: String,
    predizer: F,
) -> MetricasModelo
where
    F: Fn(&data::loader::Registro) -> f64,
{
    let n = registros.len();
    let media_real = registros.iter().map(|r| r.geracao).sum::<f64>() / n as f64;
    let dp = fis.p_max - fis.p_min;
    let mut soma_quad = 0.0;
    let mut soma_abs = 0.0;
    let mut soma_norm = 0.0;
    let mut soma_res = 0.0;
    let mut sst = 0.0;

    for r in registros {
        let pred = predizer(r);
        let erro = pred - r.geracao;
        soma_quad += erro * erro;
        soma_abs += erro.abs();
        soma_res += erro;
        soma_norm += ((pred - fis.p_min) / dp - r.geracao_norm).powi(2);
        sst += (r.geracao - media_real).powi(2);
    }

    let mse = soma_quad / n as f64;
    MetricasModelo {
        modelo,
        periodo,
        n,
        mse_norm: soma_norm / n as f64,
        rmse_mwmed: mse.sqrt(),
        mae_mwmed: soma_abs / n as f64,
        r2: if sst > 0.0 {
            1.0 - soma_quad / sst
        } else {
            f64::NAN
        },
        bias_mwmed: soma_res / n as f64,
        satura,
    }
}

fn treinar_mlp_dados(
    rede: &mut Rede,
    dados: &[data::loader::AmostraModelo],
    lr: f64,
    epocas: usize,
) -> Vec<Termos> {
    let lote: Vec<(Vec<f64>, Vec<f64>)> = dados
        .iter()
        .map(|a| (a.entrada.clone(), vec![a.alvo]))
        .collect();
    let mut adam = Adam::novo(rede, lr);
    let mut historico = Vec::with_capacity(epocas + 1);

    for epoca in 0..=epocas {
        let (grads, loss) = gradientes_lote(rede, &lote);
        historico.push(Termos {
            dados: loss,
            edp: 0.0,
            contorno: 0.0,
            total: loss,
        });
        if epoca < epocas {
            adam.passo(rede, &grads);
        }
    }

    historico
}

fn amostras_residuo_persistencia(
    registros: &[data::loader::Registro],
    fis: &Fisica,
) -> Vec<data::loader::AmostraModelo> {
    let dp = fis.p_max - fis.p_min;
    registros
        .iter()
        .map(|r| data::loader::AmostraModelo {
            entrada: entrada_multivariada(r),
            alvo: (r.geracao - r.geracao_lag1) / dp,
            q_norm: r.vazao_norm,
        })
        .collect()
}

fn amostras_residuo_persistencia_janela(
    registros: &[data::loader::Registro],
    norm: &NormalizadorJanela,
) -> Vec<data::loader::AmostraModelo> {
    let dp = norm.p_max - norm.p_min;
    registros
        .iter()
        .map(|r| data::loader::AmostraModelo {
            entrada: norm.entrada_multivariada(r),
            alvo: (r.geracao - r.geracao_lag1) / dp,
            q_norm: norm.q_norm(r),
        })
        .collect()
}

fn predizer_residuo_persistencia(
    rede: &Rede,
    fis: &Fisica,
    r: &data::loader::Registro,
    alpha: f64,
) -> f64 {
    let residuo_norm = prever(rede, &entrada_multivariada(r))[0];
    let residuo_mwmed = residuo_norm * (fis.p_max - fis.p_min);
    limitar_geracao(fis, r.geracao_lag1 + alpha * residuo_mwmed)
}

fn predizer_residuo_persistencia_janela(
    rede: &Rede,
    fis: &Fisica,
    norm: &NormalizadorJanela,
    r: &data::loader::Registro,
    alpha: f64,
) -> f64 {
    let entrada = norm.entrada_multivariada(r);
    let residuo_norm = prever(rede, &entrada)[0];
    let residuo_mwmed = residuo_norm * (fis.p_max - fis.p_min);
    limitar_geracao(fis, r.geracao_lag1 + alpha * residuo_mwmed)
}

fn selecionar_alpha_residual<F>(
    validacao: &[data::loader::Registro],
    fis: &Fisica,
    residuo_mwmed: F,
) -> (f64, Vec<(f64, f64)>)
where
    F: Fn(&data::loader::Registro) -> f64,
{
    let mut linhas = Vec::new();
    let mut melhor_alpha = 0.0;
    let mut melhor_rmse = f64::INFINITY;

    for i in 0..=20 {
        let alpha = i as f64 * 0.05;
        let (rmse, _, _) = calcular_metricas_mw(validacao, |r| {
            limitar_geracao(fis, r.geracao_lag1 + alpha * residuo_mwmed(r))
        });
        linhas.push((alpha, rmse));
        if rmse < melhor_rmse {
            melhor_rmse = rmse;
            melhor_alpha = alpha;
        }
    }

    (melhor_alpha, linhas)
}

fn salvar_selecao_residual(linhas: &[(f64, f64)], alpha_escolhido: f64) {
    let caminho = Path::new(DIR_OUTPUTS).join("selecao_residual_persistencia.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record(["alpha", "rmse_validacao_mwmed", "escolhido"]);
        for (alpha, rmse) in linhas {
            let _ = w.write_record(&[
                format!("{alpha:.2}"),
                format!("{rmse:.2}"),
                ((*alpha - alpha_escolhido).abs() < 1e-12).to_string(),
            ]);
        }
        let _ = w.flush();
    }
}

fn media_entrada_multivariada(registros: &[data::loader::Registro]) -> Vec<f64> {
    let n = registros.len() as f64;
    let mut soma = vec![0.0; 5];
    for r in registros {
        let entrada = entrada_multivariada(r);
        for (i, v) in entrada.iter().enumerate() {
            soma[i] += v;
        }
    }
    soma.iter_mut().for_each(|v| *v /= n);
    soma
}

fn media_entrada_hidrologica(registros: &[data::loader::Registro]) -> Vec<f64> {
    let n = registros.len() as f64;
    let mut soma = vec![0.0; 5];
    for r in registros {
        let entrada = entrada_hidrologica(r);
        for (i, v) in entrada.iter().enumerate() {
            soma[i] += v;
        }
    }
    soma.iter_mut().for_each(|v| *v /= n);
    soma
}

fn curva_rede(
    rede: &Rede,
    fis: &Fisica,
    n_pontos: usize,
    entrada_base: &[f64],
) -> Vec<pinn::otimizacao::PontoOtimo> {
    let mut curva = Vec::with_capacity(n_pontos);
    for i in 0..n_pontos {
        let q_norm = i as f64 / (n_pontos - 1) as f64;
        let mut entrada = entrada_base.to_vec();
        entrada[0] = q_norm;
        let p_norm = prever(rede, &entrada)[0];
        let (vazao, geracao) = fis.desnormalizar(q_norm, p_norm);
        let geracao_otima = geracao.min(fis.p_max).max(0.0);
        let eficiencia = if vazao > 1.0 {
            geracao_otima / vazao
        } else {
            0.0
        };
        curva.push(pinn::otimizacao::PontoOtimo {
            vazao,
            geracao_otima,
            eficiencia,
        });
    }
    curva
}

fn joelho_saturacao_bruta(
    rede: &Rede,
    fis: &Fisica,
    n_pontos: usize,
    entrada_base: &[f64],
) -> Option<f64> {
    for i in 0..n_pontos {
        let q_norm = i as f64 / (n_pontos - 1) as f64;
        let mut entrada = entrada_base.to_vec();
        entrada[0] = q_norm;
        let geracao_bruta = predizer_rede_bruta(rede, fis, &entrada);
        if geracao_bruta >= 0.99 * fis.p_max {
            let vazao = fis.q_min + q_norm * (fis.q_max - fis.q_min);
            return Some(vazao);
        }
    }
    None
}

fn salvar_curva_csv(nome: &str, curva: &[pinn::otimizacao::PontoOtimo]) {
    if let Ok(mut w) = csv::Writer::from_path(Path::new(DIR_OUTPUTS).join(nome)) {
        let _ = w.write_record(["vazao", "geracao_referencia", "eficiencia"]);
        for p in curva {
            let _ = w.write_record(&[
                p.vazao.to_string(),
                p.geracao_otima.to_string(),
                p.eficiencia.to_string(),
            ]);
        }
        let _ = w.flush();
    }
}

fn diagnostico_ponto(
    rede: &Rede,
    fis: &Fisica,
    entrada: &[f64],
    q_norm: f64,
) -> (f64, f64, f64, f64) {
    let geracao_bruta = predizer_rede_bruta(rede, fis, entrada);
    let geracao_pos = limitar_geracao(fis, geracao_bruta);

    let mut entrada_mais = entrada.to_vec();
    let mut entrada_menos = entrada.to_vec();
    entrada_mais[0] = q_norm + H_FD_DIAGNOSTICO;
    entrada_menos[0] = q_norm - H_FD_DIAGNOSTICO;
    let np = prever(rede, &entrada_mais)[0];
    let nm = prever(rede, &entrada_menos)[0];
    let deriv_norm = (np - nm) / (2.0 * H_FD_DIAGNOSTICO);
    let deriv_mw_por_m3s = deriv_norm * (fis.p_max - fis.p_min) / (fis.q_max - fis.q_min);
    let alvo_mw_por_m3s =
        fis.inclinacao_alvo(q_norm) * (fis.p_max - fis.p_min) / (fis.q_max - fis.q_min);

    (
        geracao_bruta,
        geracao_pos,
        deriv_mw_por_m3s,
        alvo_mw_por_m3s,
    )
}

fn resumir_diagnostico<I>(
    modelo: &'static str,
    conjunto: &'static str,
    fis: &Fisica,
    pontos: I,
) -> DiagnosticoFisico
where
    I: IntoIterator<Item = (f64, f64, f64, f64)>,
{
    let mut n = 0usize;
    let mut cortes_superiores = 0usize;
    let mut cortes_inferiores = 0usize;
    let mut soma_corte = 0.0;
    let mut corte_max = 0.0_f64;
    let mut viol_monotonia = 0usize;
    let mut soma_viol_monotonia = 0.0;
    let mut soma_abs_deriv = 0.0;
    let mut soma_quad_deriv = 0.0;
    let mut soma_bias_deriv = 0.0;

    for (geracao_bruta, geracao_pos, deriv_mw_por_m3s, alvo_mw_por_m3s) in pontos {
        n += 1;
        if geracao_bruta > fis.p_max {
            cortes_superiores += 1;
        }
        if geracao_bruta < 0.0 {
            cortes_inferiores += 1;
        }
        let corte = (geracao_bruta - geracao_pos).abs();
        soma_corte += corte;
        corte_max = corte_max.max(corte);
        if deriv_mw_por_m3s < -1e-6 {
            viol_monotonia += 1;
            soma_viol_monotonia += -deriv_mw_por_m3s;
        }
        let resid = deriv_mw_por_m3s - alvo_mw_por_m3s;
        soma_abs_deriv += resid.abs();
        soma_quad_deriv += resid * resid;
        soma_bias_deriv += resid;
    }

    let nf = n as f64;
    DiagnosticoFisico {
        modelo,
        conjunto,
        n,
        cortes_superiores,
        cortes_inferiores,
        pct_cortado: 100.0 * (cortes_superiores + cortes_inferiores) as f64 / nf,
        corte_medio_mwmed: soma_corte / nf,
        corte_max_mwmed: corte_max,
        viol_monotonia_pct: 100.0 * viol_monotonia as f64 / nf,
        viol_monotonia_media_mw_por_m3s: soma_viol_monotonia / nf,
        derivada_mae_mw_por_m3s: soma_abs_deriv / nf,
        derivada_rmse_mw_por_m3s: (soma_quad_deriv / nf).sqrt(),
        derivada_bias_mw_por_m3s: soma_bias_deriv / nf,
    }
}

fn diagnosticar_registros<F>(
    modelo: &'static str,
    conjunto: &'static str,
    rede: &Rede,
    fis: &Fisica,
    registros: &[data::loader::Registro],
    entrada: F,
) -> DiagnosticoFisico
where
    F: Fn(&data::loader::Registro) -> Vec<f64>,
{
    let pontos = registros.iter().map(|r| {
        let entrada = entrada(r);
        diagnostico_ponto(rede, fis, &entrada, r.vazao_norm)
    });
    resumir_diagnostico(modelo, conjunto, fis, pontos)
}

fn diagnosticar_curva(
    modelo: &'static str,
    rede: &Rede,
    fis: &Fisica,
    entrada_base: &[f64],
    n_pontos: usize,
) -> DiagnosticoFisico {
    let pontos = (0..n_pontos).map(|i| {
        let q_norm = i as f64 / (n_pontos - 1) as f64;
        let mut entrada = entrada_base.to_vec();
        entrada[0] = q_norm;
        diagnostico_ponto(rede, fis, &entrada, q_norm)
    });
    resumir_diagnostico(modelo, "curva_grid", fis, pontos)
}

fn salvar_diagnosticos_fisicos(linhas: &[DiagnosticoFisico]) {
    let caminho = Path::new(DIR_OUTPUTS).join("diagnostico_fisico.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "modelo",
            "conjunto",
            "n",
            "cortes_superiores",
            "cortes_inferiores",
            "pct_cortado",
            "corte_medio_mwmed",
            "corte_max_mwmed",
            "viol_monotonia_pct",
            "viol_monotonia_media_mw_por_m3s",
            "derivada_mae_mw_por_m3s",
            "derivada_rmse_mw_por_m3s",
            "derivada_bias_mw_por_m3s",
        ]);
        for d in linhas {
            let _ = w.write_record(&[
                d.modelo.to_string(),
                d.conjunto.to_string(),
                d.n.to_string(),
                d.cortes_superiores.to_string(),
                d.cortes_inferiores.to_string(),
                format!("{:.4}", d.pct_cortado),
                format!("{:.4}", d.corte_medio_mwmed),
                format!("{:.4}", d.corte_max_mwmed),
                format!("{:.4}", d.viol_monotonia_pct),
                format!("{:.6}", d.viol_monotonia_media_mw_por_m3s),
                format!("{:.6}", d.derivada_mae_mw_por_m3s),
                format!("{:.6}", d.derivada_rmse_mw_por_m3s),
                format!("{:.6}", d.derivada_bias_mw_por_m3s),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_curva_diagnostico_modelo(
    w: &mut csv::Writer<std::fs::File>,
    modelo: &'static str,
    rede: &Rede,
    fis: &Fisica,
    entrada_base: &[f64],
    n_pontos: usize,
) {
    for i in 0..n_pontos {
        let q_norm = i as f64 / (n_pontos - 1) as f64;
        let mut entrada = entrada_base.to_vec();
        entrada[0] = q_norm;
        let (geracao_bruta, geracao_pos, derivada, alvo_derivada) =
            diagnostico_ponto(rede, fis, &entrada, q_norm);
        let vazao = fis.q_min + q_norm * (fis.q_max - fis.q_min);
        let _ = w.write_record(&[
            modelo.to_string(),
            q_norm.to_string(),
            vazao.to_string(),
            geracao_bruta.to_string(),
            geracao_pos.to_string(),
            (geracao_bruta - geracao_pos).abs().to_string(),
            derivada.to_string(),
            alvo_derivada.to_string(),
        ]);
    }
}

fn salvar_curvas_diagnostico_fisico(
    modelos: &[(&'static str, &Rede, Vec<f64>)],
    fis: &Fisica,
    n_pontos: usize,
) {
    let caminho = Path::new(DIR_OUTPUTS).join("curvas_diagnostico_fisico.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "modelo",
            "q_norm",
            "vazao",
            "geracao_bruta",
            "geracao_pos_processada",
            "corte_mwmed",
            "derivada_mw_por_m3s",
            "derivada_alvo_mw_por_m3s",
        ]);
        for (modelo, rede, entrada_base) in modelos {
            salvar_curva_diagnostico_modelo(&mut w, modelo, rede, fis, entrada_base, n_pontos);
        }
        let _ = w.flush();
    }
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        f64::NAN
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

fn salvar_diagnostico_vazao_turbinada_equivalente(
    treino: &[data::loader::Registro],
    teste: &[data::loader::Registro],
    fis: &Fisica,
) {
    let k = neural::loss::k_fisico_mw_por_m3s();
    let q_capacidade = fis.p_max / k;

    let caminho_diario = Path::new(DIR_OUTPUTS).join("diagnostico_vazao_turbinada_equivalente.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho_diario) {
        let _ = w.write_record([
            "data",
            "periodo",
            "vazao_proxy_m3s",
            "geracao_real_mwmed",
            "vazao_turbinada_equivalente_m3s",
            "razao_qeq_qproxy",
            "produtividade_proxy_mw_por_m3s",
        ]);
        for (periodo, regs) in [("treino", treino), ("teste", teste)] {
            for r in regs {
                let q_eq = r.geracao / k;
                let razao = if r.vazao.abs() > 1e-12 {
                    q_eq / r.vazao
                } else {
                    f64::NAN
                };
                let produtividade_proxy = if r.vazao.abs() > 1e-12 {
                    r.geracao / r.vazao
                } else {
                    f64::NAN
                };
                let _ = w.write_record(&[
                    r.data.clone(),
                    periodo.to_string(),
                    format!("{:.4}", r.vazao),
                    format!("{:.4}", r.geracao),
                    format!("{:.4}", q_eq),
                    format!("{:.6}", razao),
                    format!("{:.6}", produtividade_proxy),
                ]);
            }
        }
        let _ = w.flush();
    }

    let treino_refs: Vec<&data::loader::Registro> = treino.iter().collect();
    let teste_refs: Vec<&data::loader::Registro> = teste.iter().collect();
    let mut total_refs = treino_refs.clone();
    total_refs.extend(teste_refs.iter().copied());

    let caminho_resumo = Path::new(DIR_OUTPUTS).join("resumo_vazao_turbinada_equivalente.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho_resumo) {
        let _ = w.write_record([
            "periodo",
            "n",
            "k_mw_por_m3s",
            "q_capacidade_equivalente_m3s",
            "vazao_proxy_media_m3s",
            "vazao_turbinada_eq_media_m3s",
            "diferenca_media_qeq_menos_proxy_m3s",
            "razao_media_qeq_proxy",
            "produtividade_proxy_media_mw_por_m3s",
            "pearson_proxy_qeq",
        ]);
        for (periodo, regs) in [
            ("treino_2015_2022", treino_refs.as_slice()),
            ("teste_2023_2024", teste_refs.as_slice()),
            ("total_2015_2024", total_refs.as_slice()),
        ] {
            let n = regs.len() as f64;
            let vazoes: Vec<f64> = regs.iter().map(|r| r.vazao).collect();
            let qeqs: Vec<f64> = regs.iter().map(|r| r.geracao / k).collect();
            let proxy_media = vazoes.iter().sum::<f64>() / n;
            let qeq_media = qeqs.iter().sum::<f64>() / n;
            let razao_media = regs.iter().map(|r| (r.geracao / k) / r.vazao).sum::<f64>() / n;
            let produtividade_proxy_media =
                regs.iter().map(|r| r.geracao / r.vazao).sum::<f64>() / n;
            let corr = pearson(&vazoes, &qeqs);
            let _ = w.write_record(&[
                periodo.to_string(),
                regs.len().to_string(),
                format!("{:.6}", k),
                format!("{:.2}", q_capacidade),
                format!("{:.2}", proxy_media),
                format!("{:.2}", qeq_media),
                format!("{:.2}", qeq_media - proxy_media),
                format!("{:.4}", razao_media),
                format!("{:.6}", produtividade_proxy_media),
                format!("{:.4}", corr),
            ]);
        }
        let _ = w.flush();
    }

    println!("[diag vazao] Q_turbinada equivalente=P/k salvo; q_capacidade≈{q_capacidade:.0} m³/s");
}

fn salvar_benchmarks_rust(linhas: &[BenchmarkModelo]) {
    let caminho = Path::new(DIR_OUTPUTS).join("benchmark_rust.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "modelo",
            "n_treino",
            "n_teste",
            "tempo_treino_s",
            "tempo_inferencia_teste_ms",
        ]);
        for b in linhas {
            let _ = w.write_record(&[
                b.modelo.to_string(),
                b.n_treino.to_string(),
                b.n_teste.to_string(),
                format!("{:.6}", b.treino_s),
                format!("{:.6}", b.inferencia_teste_ms),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_validacao_janelas(linhas: &[MetricaJanela]) {
    let caminho = Path::new(DIR_OUTPUTS).join("validacao_janelas.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "modelo",
            "ano_teste",
            "n_treino",
            "n_teste",
            "rmse_mwmed",
            "mae_mwmed",
            "r2",
            "tempo_treino_s",
        ]);
        for m in linhas {
            let _ = w.write_record(&[
                m.modelo.to_string(),
                m.ano_teste.to_string(),
                m.n_treino.to_string(),
                m.n_teste.to_string(),
                format!("{:.2}", m.rmse_mwmed),
                format!("{:.2}", m.mae_mwmed),
                format!("{:.4}", m.r2),
                format!("{:.6}", m.tempo_treino_s),
            ]);
        }
        let _ = w.flush();
    }
}

fn validacao_janelas_moveis(registros: &[data::loader::Registro], lambda_fisica: f64) {
    use pinn::treinamento;

    let lam_pinn = Lambdas {
        dados: 1.0,
        edp: lambda_fisica,
        contorno: 1.0,
    };
    let params_rf = ParametrosFloresta {
        n_arvores: 60,
        max_depth: 8,
        min_samples_split: 30,
        min_samples_leaf: 12,
        mtry: 3,
        seed: 101,
    };

    let mut linhas = Vec::new();
    println!(
        "\n[janelas] validação temporal móvel anual ({} épocas nos modelos neurais)",
        EPOCAS_JANELAS_MOVEIS
    );

    for ano_teste in 2020..=2024 {
        let treino: Vec<data::loader::Registro> = registros
            .iter()
            .filter(|r| ano_data(&r.data) < ano_teste)
            .cloned()
            .collect();
        let teste: Vec<data::loader::Registro> = registros
            .iter()
            .filter(|r| ano_data(&r.data) == ano_teste)
            .cloned()
            .collect();
        if treino.is_empty() || teste.is_empty() {
            continue;
        }

        let norm = NormalizadorJanela::ajustar(&treino);
        let fis = norm.fisica();
        let dados_multi = amostras_multivariadas_janela(&treino, &norm);
        let n_treino = treino.len();
        let n_teste = teste.len();

        let media_mensal = MediaMensal::ajustar(&treino);
        let (rmse, mae, r2) = calcular_metricas_mw(&teste, |r| media_mensal.prever(&r.data));
        linhas.push(MetricaJanela {
            modelo: "media_mensal_treino",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: 0.0,
        });

        let inicio = Instant::now();
        let linear = LinearPlato::ajustar(&treino, fis.p_max);
        let tempo_linear = inicio.elapsed().as_secs_f64();
        let (rmse, mae, r2) = calcular_metricas_mw(&teste, |r| linear.prever(r.vazao));
        linhas.push(MetricaJanela {
            modelo: "linear_com_plato",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: tempo_linear,
        });

        let (rmse, mae, r2) = calcular_metricas_mw(&teste, |r| r.geracao_lag1);
        linhas.push(MetricaJanela {
            modelo: "persistencia_dia_anterior",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: 0.0,
        });

        let ano_validacao_residual = ano_teste - 1;
        let treino_residual_interno: Vec<data::loader::Registro> = treino
            .iter()
            .filter(|r| ano_data(&r.data) < ano_validacao_residual)
            .cloned()
            .collect();
        let validacao_residual: Vec<data::loader::Registro> = treino
            .iter()
            .filter(|r| ano_data(&r.data) == ano_validacao_residual)
            .cloned()
            .collect();
        let alpha_residual = if !treino_residual_interno.is_empty()
            && !validacao_residual.is_empty()
        {
            let norm_interno = NormalizadorJanela::ajustar(&treino_residual_interno);
            let fis_interno = norm_interno.fisica();
            let dados_residual_interno =
                amostras_residuo_persistencia_janela(&treino_residual_interno, &norm_interno);
            let mut rede_alpha = Rede::nova(&[5, 16, 16, 1], 400 + ano_teste as u64);
            let _ = treinar_mlp_dados(
                &mut rede_alpha,
                &dados_residual_interno,
                0.01,
                EPOCAS_JANELAS_MOVEIS,
            );
            let (alpha, _) = selecionar_alpha_residual(&validacao_residual, &fis_interno, |r| {
                let entrada = norm_interno.entrada_multivariada(r);
                prever(&rede_alpha, &entrada)[0] * (fis_interno.p_max - fis_interno.p_min)
            });
            alpha
        } else {
            0.0
        };

        let dados_residual = amostras_residuo_persistencia_janela(&treino, &norm);
        let mut residual = Rede::nova(&[5, 16, 16, 1], 500 + ano_teste as u64);
        let inicio = Instant::now();
        let _ = treinar_mlp_dados(&mut residual, &dados_residual, 0.01, EPOCAS_JANELAS_MOVEIS);
        let tempo_residual = inicio.elapsed().as_secs_f64();
        let (rmse, mae, r2) = calcular_metricas_mw(&teste, |r| {
            predizer_residuo_persistencia_janela(&residual, &fis, &norm, r, alpha_residual)
        });
        linhas.push(MetricaJanela {
            modelo: "persistencia_residual_mlp",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: tempo_residual,
        });

        let mut mlp = Rede::nova(&[5, 16, 16, 1], 200 + ano_teste as u64);
        let inicio = Instant::now();
        let _ = treinar_mlp_dados(&mut mlp, &dados_multi, 0.01, EPOCAS_JANELAS_MOVEIS);
        let tempo_mlp = inicio.elapsed().as_secs_f64();
        let (rmse, mae, r2) =
            calcular_metricas_mw(&teste, |r| predizer_rede_janela(&mlp, &fis, &norm, r));
        linhas.push(MetricaJanela {
            modelo: "mlp_multivariado",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: tempo_mlp,
        });

        let mut pinn = Rede::nova(&[5, 16, 16, 1], 300 + ano_teste as u64);
        let inicio = Instant::now();
        let _ = treinamento::treinar(
            &mut pinn,
            &dados_multi,
            &fis,
            &lam_pinn,
            0.01,
            EPOCAS_JANELAS_MOVEIS,
            false,
        );
        let tempo_pinn = inicio.elapsed().as_secs_f64();
        let (rmse, mae, r2) =
            calcular_metricas_mw(&teste, |r| predizer_rede_janela(&pinn, &fis, &norm, r));
        linhas.push(MetricaJanela {
            modelo: "pinn_multivariada",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: tempo_pinn,
        });

        let amostras_rf = amostras_arvore_multivariadas_janela(&treino, &norm);
        let inicio = Instant::now();
        let rf = RandomForestRegressor::treinar(
            &amostras_rf,
            ParametrosFloresta {
                seed: params_rf.seed + ano_teste as u64,
                ..params_rf
            },
        );
        let tempo_rf = inicio.elapsed().as_secs_f64();
        let (rmse, mae, r2) =
            calcular_metricas_mw(&teste, |r| predizer_floresta_janela(&rf, &fis, &norm, r));
        linhas.push(MetricaJanela {
            modelo: "rf_multivariada",
            ano_teste,
            n_treino,
            n_teste,
            rmse_mwmed: rmse,
            mae_mwmed: mae,
            r2,
            tempo_treino_s: tempo_rf,
        });

        println!(
            "[janelas] ano {ano_teste}: treino={n_treino} teste={n_teste} | residual={:.0} MLP={:.0} RF={:.0} PINN={:.0} MWmed",
            linhas
                .iter()
                .rev()
                .find(|m| m.modelo == "persistencia_residual_mlp")
                .map(|m| m.rmse_mwmed)
                .unwrap_or(f64::NAN),
            linhas
                .iter()
                .rev()
                .find(|m| m.modelo == "mlp_multivariado")
                .map(|m| m.rmse_mwmed)
                .unwrap_or(f64::NAN),
            linhas
                .iter()
                .rev()
                .find(|m| m.modelo == "rf_multivariada")
                .map(|m| m.rmse_mwmed)
                .unwrap_or(f64::NAN),
            linhas
                .iter()
                .rev()
                .find(|m| m.modelo == "pinn_multivariada")
                .map(|m| m.rmse_mwmed)
                .unwrap_or(f64::NAN)
        );
    }

    salvar_validacao_janelas(&linhas);
    println!("[janelas] artefato: validacao_janelas.csv");
}

fn salvar_selecao_lambda(linhas: &[(f64, f64, f64, Option<f64>, bool)]) {
    let caminho = Path::new(DIR_OUTPUTS).join("selecao_lambda.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "lambda_fisica",
            "rmse_validacao_mwmed",
            "loss_fisica_treino",
            "joelho_saturacao_m3s",
            "elegivel",
        ]);
        for (lambda, rmse, loss_fisica, joelho, elegivel) in linhas {
            let j = joelho
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "nao_satura".into());
            let _ = w.write_record(&[
                lambda.to_string(),
                format!("{rmse:.2}"),
                format!("{loss_fisica:.6}"),
                j,
                elegivel.to_string(),
            ]);
        }
        let _ = w.flush();
    }
}

fn salvar_sensibilidade_lambda_sementes(linhas: &[SensibilidadeLambda]) {
    let caminho = Path::new(DIR_OUTPUTS).join("sensibilidade_lambda_sementes.csv");
    if let Ok(mut w) = csv::Writer::from_path(caminho) {
        let _ = w.write_record([
            "lambda_fisica",
            "seed",
            "rmse_validacao_mwmed",
            "loss_fisica_treino",
            "joelho_saturacao_m3s",
            "elegivel",
        ]);
        for linha in linhas {
            let joelho = linha
                .joelho_saturacao_m3s
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "nao_satura".to_string());
            let _ = w.write_record(&[
                linha.lambda_fisica.to_string(),
                linha.seed.to_string(),
                format!("{:.2}", linha.rmse_validacao_mwmed),
                format!("{:.6}", linha.loss_fisica_treino),
                joelho,
                linha.elegivel.to_string(),
            ]);
        }
        let _ = w.flush();
    }
}

fn media_desvio(valores: &[f64]) -> (f64, f64) {
    if valores.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let media = valores.iter().sum::<f64>() / valores.len() as f64;
    if valores.len() == 1 {
        return (media, 0.0);
    }
    let var = valores.iter().map(|v| (v - media).powi(2)).sum::<f64>() / (valores.len() - 1) as f64;
    (media, var.sqrt())
}

fn avaliar_sensibilidade_lambda_sementes(
    fis: &Fisica,
    registros_treino: &[data::loader::Registro],
) -> Option<f64> {
    use pinn::treinamento;

    let (treino_interno, validacao) = dividir_treino_validacao(registros_treino);
    let dados_interno = data::loader::amostras_vazao(&treino_interno);
    let lambdas = [0.2, 1.0];
    let seeds = [7_u64, 17, 31, 43, 59, 73, 89, 101, 131, 151];
    let mut linhas = Vec::new();

    println!("\n[sensibilidade λ] avaliando λ_fisica={{0.2,1.0}} com 10 sementes");

    for &lambda in &lambdas {
        for &seed in &seeds {
            let lam = Lambdas {
                dados: 1.0,
                edp: lambda,
                contorno: 1.0,
            };
            let mut rede = Rede::nova(&[1, 16, 16, 1], seed);
            let hist =
                treinamento::treinar(&mut rede, &dados_interno, fis, &lam, 0.01, 1200, false);
            let t = *hist.last().unwrap();
            let joelho = joelho_saturacao_bruta(&rede, fis, 100, &[0.0]);
            let met = calcular_metricas(
                "lambda_seed",
                "validacao_2021_2022",
                &validacao,
                fis,
                String::new(),
                |r| predizer_pinn(&rede, fis, r),
            );
            linhas.push(SensibilidadeLambda {
                lambda_fisica: lambda,
                seed,
                rmse_validacao_mwmed: met.rmse_mwmed,
                loss_fisica_treino: t.edp,
                joelho_saturacao_m3s: joelho,
                elegivel: joelho.is_some(),
            });
        }
    }

    salvar_sensibilidade_lambda_sementes(&linhas);
    let mut recomendado: Option<(f64, f64, f64)> = None;
    for &lambda in &lambdas {
        let rmses: Vec<f64> = linhas
            .iter()
            .filter(|l| (l.lambda_fisica - lambda).abs() < 1e-9)
            .map(|l| l.rmse_validacao_mwmed)
            .collect();
        let saturantes = linhas
            .iter()
            .filter(|l| (l.lambda_fisica - lambda).abs() < 1e-9 && l.elegivel)
            .count();
        let (media, desvio) = media_desvio(&rmses);
        println!(
            "[sensibilidade λ] λ_fisica={lambda:<3} | RMSE val médio={media:.0}±{desvio:.0} | saturantes={saturantes}/{}",
            seeds.len()
        );
        if recomendado
            .map(|(_, melhor_media, melhor_desvio)| {
                media < melhor_media
                    || ((media - melhor_media).abs() < 1e-9 && desvio < melhor_desvio)
            })
            .unwrap_or(true)
        {
            recomendado = Some((lambda, media, desvio));
        }
    }
    println!("[sensibilidade λ] artefato: sensibilidade_lambda_sementes.csv");
    if let Some((lambda, media, desvio)) = recomendado {
        println!(
            "[sensibilidade λ] recomendado λ_fisica={lambda} por média multissemente ({media:.0}±{desvio:.0} MWmed)"
        );
        Some(lambda)
    } else {
        None
    }
}

fn selecionar_lambda_fisica(fis: &Fisica, registros_treino: &[data::loader::Registro]) -> f64 {
    use pinn::{otimizacao, treinamento};

    let (treino_interno, validacao) = dividir_treino_validacao(registros_treino);
    let dados_interno = data::loader::amostras_vazao(&treino_interno);
    let candidatos = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0];
    let mut linhas = Vec::new();

    println!(
        "\n[seleção λ] treino interno={} (2015–2020) | validação={} (2021–2022)",
        treino_interno.len(),
        validacao.len()
    );
    for &lambda in &candidatos {
        let lam = Lambdas {
            dados: 1.0,
            edp: lambda,
            contorno: 1.0,
        };
        let mut rede = Rede::nova(&[1, 16, 16, 1], 7);
        let hist = treinamento::treinar(&mut rede, &dados_interno, fis, &lam, 0.01, 1200, false);
        let t = *hist.last().unwrap();
        let curva = otimizacao::curva_otima(&rede, fis, 100);
        let _joelho_pos_processado = joelho_saturacao(&curva, fis.p_max);
        let joelho = joelho_saturacao_bruta(&rede, fis, 100, &[0.0]);
        let met = calcular_metricas(
            "lambda_candidato",
            "validacao_2021_2022",
            &validacao,
            fis,
            String::new(),
            |r| predizer_pinn(&rede, fis, r),
        );
        let elegivel = lambda > 0.0 && joelho.is_some();
        linhas.push((lambda, met.rmse_mwmed, t.edp, joelho, elegivel));
        println!(
            "[seleção λ] λ_fisica={lambda:<4} | RMSE val={:.0} | loss_fisica={:.4} | {}",
            met.rmse_mwmed,
            t.edp,
            joelho
                .map(|v| format!("satura ~{v:.0} m³/s"))
                .unwrap_or_else(|| "não satura".into())
        );
    }

    salvar_selecao_lambda(&linhas);
    let melhor = linhas
        .iter()
        .filter(|(_, _, _, _, elegivel)| *elegivel)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .or_else(|| linhas.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()))
        .map(|(lambda, _, _, _, _)| *lambda)
        .unwrap_or(0.2);
    println!("[seleção λ] escolhido na varredura pontual λ_fisica={melhor}");
    melhor
}

fn salvar_validacao_temporal(
    rede_pinn_1d: &Rede,
    rede_mlp_1d: &Rede,
    rede_pinn_hidrologica: &Rede,
    rede_mlp_hidrologico: &Rede,
    rede_pinn_multivariada: &Rede,
    rede_mlp_multivariada: &Rede,
    rede_residual_persistencia: &Rede,
    floresta_hidrologica: &RandomForestRegressor,
    floresta_multivariada: &RandomForestRegressor,
    fis: &Fisica,
    treino: &[data::loader::Registro],
    teste: &[data::loader::Registro],
    joelho_pinn_1d: Option<f64>,
    joelho_pinn_hidrologica: Option<f64>,
    joelho_pinn_multivariada: Option<f64>,
    alpha_residual_persistencia: f64,
) {
    let media_treino = MediaConstante::ajustar(treino);
    let media_mensal = MediaMensal::ajustar(treino);
    let linear = LinearPlato::ajustar(treino, fis.p_max);
    let quadratica = QuadraticaReferencia::ajustar(treino);
    let satura_pinn_1d = joelho_pinn_1d
        .map(|v| format!("sim (~{v:.0} m3/s)"))
        .unwrap_or_else(|| "nao".to_string());
    let satura_pinn_hidro = joelho_pinn_hidrologica
        .map(|v| format!("sim (~{v:.0} m3/s)"))
        .unwrap_or_else(|| "nao".to_string());
    let satura_pinn_multi = joelho_pinn_multivariada
        .map(|v| format!("sim (~{v:.0} m3/s)"))
        .unwrap_or_else(|| "nao".to_string());
    let satura_linear = linear
        .q_sat()
        .map(|v| format!("sim (~{v:.0} m3/s)"))
        .unwrap_or_else(|| "nao".to_string());

    let mut linhas = Vec::new();
    for (nome_periodo, regs) in [("treino_2015_2022", treino), ("teste_2023_2024", teste)] {
        linhas.push(calcular_metricas(
            "media_treino",
            nome_periodo,
            regs,
            fis,
            "nao_aplica".to_string(),
            |_| media_treino.prever(),
        ));
        linhas.push(calcular_metricas(
            "media_mensal_treino",
            nome_periodo,
            regs,
            fis,
            "nao_aplica".to_string(),
            |r| media_mensal.prever(&r.data),
        ));
        linhas.push(calcular_metricas(
            "interpolacao_quadratica",
            nome_periodo,
            regs,
            fis,
            "nao".to_string(),
            |r| quadratica.prever(r.vazao),
        ));
        linhas.push(calcular_metricas(
            "linear_com_plato",
            nome_periodo,
            regs,
            fis,
            satura_linear.clone(),
            |r| linear.prever(r.vazao),
        ));
        linhas.push(calcular_metricas(
            "persistencia_dia_anterior",
            nome_periodo,
            regs,
            fis,
            "nao_aplica".to_string(),
            |r| r.geracao_lag1,
        ));
        linhas.push(calcular_metricas(
            "persistencia_residual_mlp",
            nome_periodo,
            regs,
            fis,
            format!("alpha={alpha_residual_persistencia:.2}"),
            |r| {
                predizer_residuo_persistencia(
                    rede_residual_persistencia,
                    fis,
                    r,
                    alpha_residual_persistencia,
                )
            },
        ));
        linhas.push(calcular_metricas(
            "mlp_1d",
            nome_periodo,
            regs,
            fis,
            "nao_forcada".to_string(),
            |r| predizer_pinn(rede_mlp_1d, fis, r),
        ));
        linhas.push(calcular_metricas(
            "pinn_fisica_1d",
            nome_periodo,
            regs,
            fis,
            satura_pinn_1d.clone(),
            |r| predizer_pinn(rede_pinn_1d, fis, r),
        ));
        linhas.push(calcular_metricas(
            "mlp_hidrologico",
            nome_periodo,
            regs,
            fis,
            "nao_forcada".to_string(),
            |r| predizer_hidrologica(rede_mlp_hidrologico, fis, r),
        ));
        linhas.push(calcular_metricas(
            "pinn_hidrologica",
            nome_periodo,
            regs,
            fis,
            satura_pinn_hidro.clone(),
            |r| predizer_hidrologica(rede_pinn_hidrologica, fis, r),
        ));
        linhas.push(calcular_metricas(
            "rf_hidrologica",
            nome_periodo,
            regs,
            fis,
            "nao_forcada".to_string(),
            |r| predizer_floresta(floresta_hidrologica, fis, r, entrada_hidrologica(r)),
        ));
        linhas.push(calcular_metricas(
            "mlp_multivariado",
            nome_periodo,
            regs,
            fis,
            "nao_forcada".to_string(),
            |r| predizer_multivariada(rede_mlp_multivariada, fis, r),
        ));
        linhas.push(calcular_metricas(
            "rf_multivariada",
            nome_periodo,
            regs,
            fis,
            "nao_forcada".to_string(),
            |r| predizer_floresta(floresta_multivariada, fis, r, entrada_multivariada(r)),
        ));
        linhas.push(calcular_metricas(
            "pinn_multivariada",
            nome_periodo,
            regs,
            fis,
            satura_pinn_multi.clone(),
            |r| predizer_multivariada(rede_pinn_multivariada, fis, r),
        ));
    }

    let caminho = Path::new(DIR_OUTPUTS).join("validacao_temporal.csv");
    if let Ok(mut w) = csv::Writer::from_path(&caminho) {
        let _ = w.write_record([
            "modelo",
            "periodo",
            "n",
            "mse_norm",
            "rmse_mwmed",
            "mae_mwmed",
            "r2",
            "bias_mwmed",
            "satura",
        ]);
        for m in &linhas {
            let _ = w.write_record(&[
                m.modelo.to_string(),
                m.periodo.to_string(),
                m.n.to_string(),
                format!("{:.8}", m.mse_norm),
                format!("{:.2}", m.rmse_mwmed),
                format!("{:.2}", m.mae_mwmed),
                format!("{:.4}", m.r2),
                format!("{:.2}", m.bias_mwmed),
                m.satura.clone(),
            ]);
        }
        let _ = w.flush();
    }

    let caminho_pred = Path::new(DIR_OUTPUTS).join("predicoes_validacao_temporal.csv");
    if let Ok(mut w) = csv::Writer::from_path(&caminho_pred) {
        let _ = w.write_record([
            "data",
            "periodo",
            "vazao",
            "geracao_real_mwmed",
            "pinn_fisica_1d_mwmed",
            "mlp_1d_mwmed",
            "mlp_hidrologico_mwmed",
            "pinn_hidrologica_mwmed",
            "rf_hidrologica_mwmed",
            "mlp_multivariado_mwmed",
            "rf_multivariada_mwmed",
            "pinn_multivariada_mwmed",
            "media_treino_mwmed",
            "media_mensal_treino_mwmed",
            "linear_plato_mwmed",
            "interpolacao_quadratica_mwmed",
            "persistencia_dia_anterior_mwmed",
            "persistencia_residual_mlp_mwmed",
            "residuo_pinn_multivariada_mwmed",
        ]);
        for (periodo, regs) in [("treino", treino), ("teste", teste)] {
            for r in regs {
                let pinn_1d = predizer_pinn(rede_pinn_1d, fis, r);
                let mlp_1d = predizer_pinn(rede_mlp_1d, fis, r);
                let mlp_hidro = predizer_hidrologica(rede_mlp_hidrologico, fis, r);
                let pinn_hidro = predizer_hidrologica(rede_pinn_hidrologica, fis, r);
                let rf_hidro =
                    predizer_floresta(floresta_hidrologica, fis, r, entrada_hidrologica(r));
                let mlp_multi = predizer_multivariada(rede_mlp_multivariada, fis, r);
                let rf_multi =
                    predizer_floresta(floresta_multivariada, fis, r, entrada_multivariada(r));
                let pinn_multi = predizer_multivariada(rede_pinn_multivariada, fis, r);
                let media = media_treino.prever();
                let media_mes = media_mensal.prever(&r.data);
                let lin = linear.prever(r.vazao);
                let quad = quadratica.prever(r.vazao);
                let persistencia = r.geracao_lag1;
                let residual = predizer_residuo_persistencia(
                    rede_residual_persistencia,
                    fis,
                    r,
                    alpha_residual_persistencia,
                );
                let _ = w.write_record(&[
                    r.data.clone(),
                    periodo.to_string(),
                    r.vazao.to_string(),
                    r.geracao.to_string(),
                    pinn_1d.to_string(),
                    mlp_1d.to_string(),
                    mlp_hidro.to_string(),
                    pinn_hidro.to_string(),
                    rf_hidro.to_string(),
                    mlp_multi.to_string(),
                    rf_multi.to_string(),
                    pinn_multi.to_string(),
                    media.to_string(),
                    media_mes.to_string(),
                    lin.to_string(),
                    quad.to_string(),
                    persistencia.to_string(),
                    residual.to_string(),
                    (pinn_multi - r.geracao).to_string(),
                ]);
            }
        }
        let _ = w.flush();
    }

    println!(
        "[validacao] linear com platô: P(Q)=min({:.1} + {:.4}Q, {:.0}); q_sat≈{}",
        linear.a,
        linear.b,
        linear.p_max,
        linear
            .q_sat()
            .map(|v| format!("{v:.0} m³/s"))
            .unwrap_or_else(|| "indefinido".into())
    );
    if let Some(v) = quadratica.vertice() {
        println!(
            "[validacao] interpolação quadrática usa pontos {:?}; vértice≈{v:.0} m³/s",
            quadratica.pontos
        );
    }
    for m in linhas.iter().filter(|m| m.periodo == "teste_2023_2024") {
        println!(
            "[teste] {:<26} RMSE={:>7.0} MWmed | MAE={:>7.0} | R²={:>6.3} | {}",
            m.modelo, m.rmse_mwmed, m.mae_mwmed, m.r2, m.satura
        );
    }
    println!("[validacao] artefatos: validacao_temporal.csv, predicoes_validacao_temporal.csv");
}

/// Treina a PINN nos dados reais, escolhe λ_fisica por validação interna
/// combinada com sensibilidade multissemente, roda a ablação e salva métricas.
fn fase5_pinn(
    registros_treino: &[data::loader::Registro],
    registros_teste: &[data::loader::Registro],
) -> Option<(Rede, Fisica)> {
    use pinn::{otimizacao, treinamento};

    println!(" PINN completa (dados + física)");

    let norm = match data::loader::carregar_normalizacao(CAMINHO_NORM) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("ERRO ao ler {CAMINHO_NORM}: {e}");
            return None;
        }
    };
    let fis = Fisica::nova(
        norm.vazao.min,
        norm.vazao.max,
        norm.geracao.min,
        norm.geracao.max,
    );
    println!(
        "[pinn] k={:.3} MW/(m³/s) | vazão de saturação física ≈ {:.0} m³/s",
        neural::loss::k_fisico_mw_por_m3s(),
        otimizacao::vazao_saturacao(&fis),
    );

    let _ = std::fs::create_dir_all(DIR_OUTPUTS);
    salvar_diagnostico_vazao_turbinada_equivalente(registros_treino, registros_teste, &fis);
    let lambda_pontual = selecionar_lambda_fisica(&fis, registros_treino);
    let lambda_escolhida =
        avaliar_sensibilidade_lambda_sementes(&fis, registros_treino).unwrap_or(lambda_pontual);
    println!(
        "[seleção λ] λ_fisica usado no modelo final={lambda_escolhida} | seleção pontual={lambda_pontual}"
    );
    let dados = data::loader::amostras_vazao(registros_treino);

    // Ablação de λ_fisica ∈ {0.0, 0.2, 1.0}; a escolha final vem da combinação
    // entre validação pontual e sensibilidade multissemente. Nesta ablação mantemos
    // λ_contorno=1.0 para isolar apenas o efeito do resíduo físico.
    println!("\n[ablação] efeito do termo físico (λ_fisica). λ_dados=λ_contorno=1.0");
    println!("[ablação] treino: 2015–2022; teste temporal reservado: 2023–2024");
    let mut linhas_ablacao: Vec<(f64, f64, f64, Option<f64>, f64)> = Vec::new();
    let lambda_fisicas = [0.0, 0.2, 1.0];
    let n_curva = 100;

    // vazão compartilhada (mesmo grid p/ todas) + geração de cada λ (p/ a figura sobreposta)
    let mut vazao_grid: Vec<f64> = Vec::new();
    let mut geracoes: Vec<Vec<f64>> = Vec::new();
    let mut modelo_principal: Option<Rede> = None;
    let mut joelho_principal: Option<f64> = None;
    let mut tempo_treino_pinn_1d = 0.0;

    for &le in &lambda_fisicas {
        let lam = Lambdas {
            dados: 1.0,
            edp: le,
            contorno: 1.0,
        };
        let mut rede = Rede::nova(&[1, 16, 16, 1], 7);
        let inicio_treino = Instant::now();
        let hist = treinamento::treinar(&mut rede, &dados, &fis, &lam, 0.01, 2000, false);
        let tempo_treino = inicio_treino.elapsed().as_secs_f64();
        let t = *hist.last().unwrap();
        let curva = otimizacao::curva_otima(&rede, &fis, n_curva);
        let joelho = joelho_saturacao_bruta(&rede, &fis, n_curva, &[0.0]);
        linhas_ablacao.push((le, t.dados, t.edp, joelho, t.contorno));

        if vazao_grid.is_empty() {
            vazao_grid = curva.iter().map(|p| p.vazao).collect();
        }
        geracoes.push(curva.iter().map(|p| p.geracao_otima).collect());

        let satura = match joelho {
            Some(v) => format!("satura ~{v:.0} m³/s"),
            None => "NÃO satura".to_string(),
        };
        println!(
            "[ablação] λ_fisica={le:<3} | loss_dados={:.4} loss_fisica(resíduo)={:.4} | {satura}",
            t.dados, t.edp
        );

        // O modelo principal (λ_fisica escolhido) tem seus artefatos salvos e é retornado.
        if (le - lambda_escolhida).abs() < 1e-9 {
            let _ = treinamento::salvar_historico(
                &hist,
                &Path::new(DIR_OUTPUTS).join("historico_loss.csv"),
            );
            let _ =
                otimizacao::salvar_curva(&curva, &Path::new(DIR_OUTPUTS).join("curva_otima.csv"));
            joelho_principal = joelho;
            tempo_treino_pinn_1d = tempo_treino;
            modelo_principal = Some(rede.clone());
        }
    }

    // 3 curvas sobrepostas em um único CSV
    if let Ok(mut w) = csv::Writer::from_path(Path::new(DIR_OUTPUTS).join("curvas_ablacao.csv")) {
        let _ = w.write_record(["vazao", "geracao_l0", "geracao_l02", "geracao_l1"]);
        for i in 0..vazao_grid.len() {
            let _ = w.write_record(&[
                vazao_grid[i].to_string(),
                geracoes[0][i].to_string(),
                geracoes[1][i].to_string(),
                geracoes[2][i].to_string(),
            ]);
        }
        let _ = w.flush();
    }

    let dados_multi = data::loader::amostras_multivariadas(registros_treino);
    let dados_hidro = data::loader::amostras_hidrologicas(registros_treino);
    let lam_principal = Lambdas {
        dados: 1.0,
        edp: lambda_escolhida,
        contorno: 1.0,
    };

    let mut rede_mlp_1d = Rede::nova(&[1, 16, 16, 1], 13);
    let inicio_treino = Instant::now();
    let hist_mlp_1d = treinar_mlp_dados(&mut rede_mlp_1d, &dados, 0.01, 2000);
    let tempo_treino_mlp_1d = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_mlp_1d,
        &Path::new(DIR_OUTPUTS).join("historico_loss_mlp_1d.csv"),
    );
    println!("[mlp] MLP 1D treinado apenas com loss de dados");

    let mut rede_mlp_hidrologico = Rede::nova(&[5, 16, 16, 1], 19);
    let inicio_treino = Instant::now();
    let hist_mlp_hidro = treinar_mlp_dados(&mut rede_mlp_hidrologico, &dados_hidro, 0.01, 2000);
    let tempo_treino_mlp_hidro = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_mlp_hidro,
        &Path::new(DIR_OUTPUTS).join("historico_loss_mlp_hidrologico.csv"),
    );
    let entrada_base_hidro = media_entrada_hidrologica(registros_treino);
    let curva_mlp_hidro = curva_rede(&rede_mlp_hidrologico, &fis, n_curva, &entrada_base_hidro);
    salvar_curva_csv("curva_mlp_hidrologico.csv", &curva_mlp_hidro);
    println!("[mlp] MLP hidrológico treinado sem geração defasada");

    let mut rede_mlp_multivariada = Rede::nova(&[5, 16, 16, 1], 17);
    let inicio_treino = Instant::now();
    let hist_mlp_multi = treinar_mlp_dados(&mut rede_mlp_multivariada, &dados_multi, 0.01, 2000);
    let tempo_treino_mlp_multi = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_mlp_multi,
        &Path::new(DIR_OUTPUTS).join("historico_loss_mlp_multivariado.csv"),
    );
    let entrada_base_multi = media_entrada_multivariada(registros_treino);
    let curva_mlp_multi = curva_rede(&rede_mlp_multivariada, &fis, n_curva, &entrada_base_multi);
    salvar_curva_csv("curva_mlp_multivariado.csv", &curva_mlp_multi);
    println!("[mlp] MLP multivariado treinado sem geração defasada");

    let (treino_residual_interno, validacao_residual) = dividir_treino_validacao(registros_treino);
    let dados_residual_interno = amostras_residuo_persistencia(&treino_residual_interno, &fis);
    let mut rede_residual_alpha = Rede::nova(&[5, 16, 16, 1], 41);
    let _ = treinar_mlp_dados(
        &mut rede_residual_alpha,
        &dados_residual_interno,
        0.01,
        1200,
    );
    let (alpha_residual, linhas_alpha_residual) =
        selecionar_alpha_residual(&validacao_residual, &fis, |r| {
            prever(&rede_residual_alpha, &entrada_multivariada(r))[0] * (fis.p_max - fis.p_min)
        });
    salvar_selecao_residual(&linhas_alpha_residual, alpha_residual);

    let dados_residual = amostras_residuo_persistencia(registros_treino, &fis);
    let mut rede_residual_persistencia = Rede::nova(&[5, 16, 16, 1], 43);
    let inicio_treino = Instant::now();
    let hist_residual =
        treinar_mlp_dados(&mut rede_residual_persistencia, &dados_residual, 0.01, 2000);
    let tempo_treino_residual = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_residual,
        &Path::new(DIR_OUTPUTS).join("historico_loss_residual_persistencia.csv"),
    );
    println!(
        "[residual] persistência + MLP residual sem P(t-1) nas entradas | alpha={alpha_residual:.2}"
    );

    let mut rede_pinn_hidrologica = Rede::nova(&[5, 16, 16, 1], 23);
    let inicio_treino = Instant::now();
    let hist_pinn_hidro = treinamento::treinar(
        &mut rede_pinn_hidrologica,
        &dados_hidro,
        &fis,
        &lam_principal,
        0.01,
        2000,
        false,
    );
    let tempo_treino_pinn_hidro = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_pinn_hidro,
        &Path::new(DIR_OUTPUTS).join("historico_loss_pinn_hidrologica.csv"),
    );
    let curva_pinn_hidro = curva_rede(&rede_pinn_hidrologica, &fis, n_curva, &entrada_base_hidro);
    let joelho_pinn_hidrologica =
        joelho_saturacao_bruta(&rede_pinn_hidrologica, &fis, n_curva, &entrada_base_hidro);
    salvar_curva_csv("curva_pinn_hidrologica.csv", &curva_pinn_hidro);
    println!(
        "[hidro] PINN hidrológica (sem geração defasada) | {}",
        joelho_pinn_hidrologica
            .map(|v| format!("satura ~{v:.0} m³/s"))
            .unwrap_or_else(|| "NÃO satura".into())
    );

    let mut rede_multivariada = Rede::nova(&[5, 16, 16, 1], 11);
    let inicio_treino = Instant::now();
    let hist_multi = treinamento::treinar(
        &mut rede_multivariada,
        &dados_multi,
        &fis,
        &lam_principal,
        0.01,
        2000,
        false,
    );
    let tempo_treino_pinn_multi = inicio_treino.elapsed().as_secs_f64();
    let _ = treinamento::salvar_historico(
        &hist_multi,
        &Path::new(DIR_OUTPUTS).join("historico_loss_multivariada.csv"),
    );
    let curva_multi = curva_rede(&rede_multivariada, &fis, n_curva, &entrada_base_multi);
    let joelho_multivariada =
        joelho_saturacao_bruta(&rede_multivariada, &fis, n_curva, &entrada_base_multi);
    salvar_curva_csv("curva_multivariada.csv", &curva_multi);
    println!(
        "[multi] rede física multivariada (vazão+ENA+sazonalidade+vazão_lag1) | {}",
        joelho_multivariada
            .map(|v| format!("satura ~{v:.0} m³/s"))
            .unwrap_or_else(|| "NÃO satura".into())
    );

    let params_hidro = ParametrosFloresta {
        n_arvores: 80,
        max_depth: 9,
        min_samples_split: 30,
        min_samples_leaf: 12,
        mtry: 3,
        seed: 31,
    };
    let params_multi = ParametrosFloresta {
        n_arvores: 80,
        max_depth: 9,
        min_samples_split: 30,
        min_samples_leaf: 12,
        mtry: 3,
        seed: 37,
    };

    let amostras_rf_hidro = amostras_arvore(registros_treino, entrada_hidrologica);
    let inicio_treino = Instant::now();
    let floresta_hidrologica = RandomForestRegressor::treinar(&amostras_rf_hidro, params_hidro);
    let tempo_treino_rf_hidro = inicio_treino.elapsed().as_secs_f64();
    println!("[rf] Random Forest hidrológica treinada em Rust");

    let amostras_rf_multi = amostras_arvore(registros_treino, entrada_multivariada);
    let inicio_treino = Instant::now();
    let floresta_multivariada = RandomForestRegressor::treinar(&amostras_rf_multi, params_multi);
    let tempo_treino_rf_multi = inicio_treino.elapsed().as_secs_f64();
    println!("[rf] Random Forest multivariada treinada em Rust");

    // salva a tabela de ablação
    if let Ok(mut w) = csv::Writer::from_path(Path::new(DIR_OUTPUTS).join("ablacao_lambda.csv")) {
        let _ = w.write_record([
            "lambda_fisica",
            "loss_dados",
            "loss_fisica_residuo",
            "joelho_saturacao_m3s",
            "loss_contorno",
        ]);
        for (le, ld, le2, joelho, lc) in &linhas_ablacao {
            let j = joelho
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "nao_satura".into());
            let _ = w.write_record(&[
                le.to_string(),
                ld.to_string(),
                le2.to_string(),
                j,
                lc.to_string(),
            ]);
        }
        let _ = w.flush();
    }

    if let Some(ref principal) = modelo_principal {
        let diagnosticos = vec![
            diagnosticar_registros(
                "pinn_fisica_1d",
                "teste_2023_2024",
                principal,
                &fis,
                registros_teste,
                entrada_vazao,
            ),
            diagnosticar_curva("pinn_fisica_1d", principal, &fis, &[0.0], n_curva),
            diagnosticar_registros(
                "pinn_hidrologica",
                "teste_2023_2024",
                &rede_pinn_hidrologica,
                &fis,
                registros_teste,
                entrada_hidrologica,
            ),
            diagnosticar_curva(
                "pinn_hidrologica",
                &rede_pinn_hidrologica,
                &fis,
                &entrada_base_hidro,
                n_curva,
            ),
            diagnosticar_registros(
                "pinn_multivariada",
                "teste_2023_2024",
                &rede_multivariada,
                &fis,
                registros_teste,
                entrada_multivariada,
            ),
            diagnosticar_curva(
                "pinn_multivariada",
                &rede_multivariada,
                &fis,
                &entrada_base_multi,
                n_curva,
            ),
            diagnosticar_registros(
                "mlp_multivariado",
                "teste_2023_2024",
                &rede_mlp_multivariada,
                &fis,
                registros_teste,
                entrada_multivariada,
            ),
            diagnosticar_curva(
                "mlp_multivariado",
                &rede_mlp_multivariada,
                &fis,
                &entrada_base_multi,
                n_curva,
            ),
        ];
        salvar_diagnosticos_fisicos(&diagnosticos);
        salvar_curvas_diagnostico_fisico(
            &[
                ("pinn_fisica_1d", principal, vec![0.0]),
                (
                    "pinn_hidrologica",
                    &rede_pinn_hidrologica,
                    entrada_base_hidro.clone(),
                ),
                (
                    "pinn_multivariada",
                    &rede_multivariada,
                    entrada_base_multi.clone(),
                ),
                (
                    "mlp_multivariado",
                    &rede_mlp_multivariada,
                    entrada_base_multi.clone(),
                ),
            ],
            &fis,
            n_curva,
        );
        if let Some(d) = diagnosticos
            .iter()
            .find(|d| d.modelo == "pinn_multivariada" && d.conjunto == "curva_grid")
        {
            println!(
                "[diag] PINN multivariada curva bruta: cortes={:.1}% | viol. monotonia={:.1}% | MAE derivada={:.3} MW/(m3/s)",
                d.pct_cortado, d.viol_monotonia_pct, d.derivada_mae_mw_por_m3s
            );
        }

        salvar_validacao_temporal(
            principal,
            &rede_mlp_1d,
            &rede_pinn_hidrologica,
            &rede_mlp_hidrologico,
            &rede_multivariada,
            &rede_mlp_multivariada,
            &rede_residual_persistencia,
            &floresta_hidrologica,
            &floresta_multivariada,
            &fis,
            registros_treino,
            registros_teste,
            joelho_principal,
            joelho_pinn_hidrologica,
            joelho_multivariada,
            alpha_residual,
        );
        experimentos::explorar_plato_temporal(registros_treino, registros_teste, &fis);

        let benchmarks = vec![
            BenchmarkModelo {
                modelo: "mlp_1d",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_mlp_1d,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_pinn(&rede_mlp_1d, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "pinn_fisica_1d",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_pinn_1d,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_pinn(principal, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "mlp_hidrologico",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_mlp_hidro,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_hidrologica(&rede_mlp_hidrologico, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "pinn_hidrologica",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_pinn_hidro,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_hidrologica(&rede_pinn_hidrologica, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "rf_hidrologica",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_rf_hidro,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_floresta(&floresta_hidrologica, &fis, r, entrada_hidrologica(r))
                }),
            },
            BenchmarkModelo {
                modelo: "mlp_multivariado",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_mlp_multi,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_multivariada(&rede_mlp_multivariada, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "persistencia_residual_mlp",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_residual,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_residuo_persistencia(
                        &rede_residual_persistencia,
                        &fis,
                        r,
                        alpha_residual,
                    )
                }),
            },
            BenchmarkModelo {
                modelo: "pinn_multivariada",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_pinn_multi,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_multivariada(&rede_multivariada, &fis, r)
                }),
            },
            BenchmarkModelo {
                modelo: "rf_multivariada",
                n_treino: registros_treino.len(),
                n_teste: registros_teste.len(),
                treino_s: tempo_treino_rf_multi,
                inferencia_teste_ms: medir_inferencia(registros_teste, |r| {
                    predizer_floresta(&floresta_multivariada, &fis, r, entrada_multivariada(r))
                }),
            },
        ];
        salvar_benchmarks_rust(&benchmarks);
    }

    let mut registros_janelas = registros_treino.to_vec();
    registros_janelas.extend_from_slice(registros_teste);
    validacao_janelas_moveis(&registros_janelas, lambda_escolhida);

    println!("\n[pinn] artefatos: historico_loss.csv, curva_otima.csv, ablacao_lambda.csv, curvas_ablacao.csv");
    println!(
        "✓ Fase 5 OK — PINN treinada, ablação e validação temporal concluídas (pronta p/ Fase 6)."
    );

    modelo_principal.map(|rede| (rede, fis))
}

/// Compara, sobre o histórico, a geração real vs. a curva de referência da PINN.
///
/// Enquadramento (decisão metodológica — ver docs/decisoes_tecnicas.md §11):
/// a curva da PINN é uma referência física-informada treinada em 2015–2022. Como
/// Itaipu tem reservatório (vazão↔geração desacopladas), não se reivindica
/// ganho financeiro nem prova de eficiência operacional a partir dos resíduos.
fn fase6_retrospectiva(
    rede: &Rede,
    fis: &Fisica,
    registros_treino: &[data::loader::Registro],
    registros_teste: &[data::loader::Registro],
) {
    use simulacao::retrospectiva;

    println!("análise retrospectiva temporal (real vs. referência)");

    let (mut dias, resumo_treino) = retrospectiva::simular(rede, fis, registros_treino);
    let (dias_teste, resumo_teste) = retrospectiva::simular(rede, fis, registros_teste);
    dias.extend(dias_teste);

    let caminho = Path::new(DIR_OUTPUTS).join("simulacao_retrospectiva.csv");
    if let Err(e) = retrospectiva::salvar(&dias, &caminho) {
        eprintln!("ERRO ao salvar simulação: {e}");
        return;
    }

    fn imprimir_resumo(rotulo: &str, resumo: &simulacao::retrospectiva::Resumo) {
        let twh = |mwh: f64| mwh / 1.0e6; // MWh -> TWh
        let delta_neg = resumo.delta_mwh_liquido - resumo.delta_mwh_positivo;
        let pct_liquido = 100.0 * resumo.delta_mwh_liquido / resumo.energia_real_mwh;
        let dias_sobre = resumo.n_dias - resumo.dias_com_ganho;

        println!("[sim:{rotulo}] dias analisados: {}", resumo.n_dias);
        println!(
            "[sim:{rotulo}] energia real gerada: {:.0} TWh",
            twh(resumo.energia_real_mwh)
        );
        println!(
            "[sim:{rotulo}] resíduo líquido (referência PINN - real): {:.1} TWh = {:+.1}%",
            twh(resumo.delta_mwh_liquido),
            pct_liquido
        );
        println!(
            "      abaixo da referência: {:.1}% dos dias (+{:.1} TWh) | acima: {:.1}% ({:.1} TWh)",
            100.0 * resumo.dias_com_ganho as f64 / resumo.n_dias as f64,
            twh(resumo.delta_mwh_positivo),
            100.0 * dias_sobre as f64 / resumo.n_dias as f64,
            twh(delta_neg)
        );
    }

    imprimir_resumo("treino 2015–2022", &resumo_treino);
    imprimir_resumo("teste 2023–2024", &resumo_teste);
    println!("[sim] conclusão: análise temporal de resíduos; sem ganho sistemático a reivindicar.");

    println!("\n[sim] salvo em data/outputs/simulacao_retrospectiva.csv");
}
