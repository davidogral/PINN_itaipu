#[derive(Debug, Clone)]
pub struct AmostraArvore {
    pub x: Vec<f64>,
    pub y: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ParametrosFloresta {
    pub n_arvores: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub mtry: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
enum No {
    Folha {
        valor: f64,
    },
    Divisao {
        feature: usize,
        limiar: f64,
        esquerda: Box<No>,
        direita: Box<No>,
    },
}

#[derive(Debug, Clone)]
pub struct ArvoreRegressao {
    raiz: No,
}

#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    arvores: Vec<ArvoreRegressao>,
}

#[derive(Debug, Clone, Copy)]
struct MelhorDivisao {
    feature: usize,
    limiar: f64,
    sse: f64,
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    estado: u64,
}

impl XorShift64 {
    fn novo(seed: u64) -> Self {
        XorShift64 {
            estado: seed.max(1),
        }
    }

    fn proximo_u64(&mut self) -> u64 {
        let mut x = self.estado;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.estado = x;
        x
    }

    fn usize(&mut self, limite: usize) -> usize {
        if limite == 0 {
            0
        } else {
            (self.proximo_u64() as usize) % limite
        }
    }
}

impl ArvoreRegressao {
    pub fn treinar(amostras: &[AmostraArvore], params: &ParametrosFloresta, seed: u64) -> Self {
        let indices: Vec<usize> = (0..amostras.len()).collect();
        let mut rng = XorShift64::novo(seed);
        let raiz = Self::construir(amostras, &indices, params, 0, &mut rng);
        ArvoreRegressao { raiz }
    }

    fn construir(
        amostras: &[AmostraArvore],
        indices: &[usize],
        params: &ParametrosFloresta,
        depth: usize,
        rng: &mut XorShift64,
    ) -> No {
        let valor = media(amostras, indices);
        if depth >= params.max_depth
            || indices.len() < params.min_samples_split
            || variancia(amostras, indices, valor) < 1e-12
        {
            return No::Folha { valor };
        }

        let n_features = amostras[0].x.len();
        let features = escolher_features(n_features, params.mtry.max(1).min(n_features), rng);
        let Some(melhor) = melhor_divisao(amostras, indices, &features, params.min_samples_leaf)
        else {
            return No::Folha { valor };
        };

        let mut esquerda = Vec::new();
        let mut direita = Vec::new();
        for &idx in indices {
            if amostras[idx].x[melhor.feature] <= melhor.limiar {
                esquerda.push(idx);
            } else {
                direita.push(idx);
            }
        }

        if esquerda.len() < params.min_samples_leaf || direita.len() < params.min_samples_leaf {
            return No::Folha { valor };
        }

        No::Divisao {
            feature: melhor.feature,
            limiar: melhor.limiar,
            esquerda: Box::new(Self::construir(amostras, &esquerda, params, depth + 1, rng)),
            direita: Box::new(Self::construir(amostras, &direita, params, depth + 1, rng)),
        }
    }

    pub fn prever(&self, x: &[f64]) -> f64 {
        prever_no(&self.raiz, x)
    }
}

impl RandomForestRegressor {
    pub fn treinar(amostras: &[AmostraArvore], params: ParametrosFloresta) -> Self {
        let mut rng = XorShift64::novo(params.seed);
        let mut arvores = Vec::with_capacity(params.n_arvores);
        for _ in 0..params.n_arvores {
            let mut bootstrap = Vec::with_capacity(amostras.len());
            for _ in 0..amostras.len() {
                bootstrap.push(amostras[rng.usize(amostras.len())].clone());
            }
            let seed_arvore = rng.proximo_u64();
            arvores.push(ArvoreRegressao::treinar(&bootstrap, &params, seed_arvore));
        }
        RandomForestRegressor { arvores }
    }

    pub fn prever(&self, x: &[f64]) -> f64 {
        self.arvores.iter().map(|a| a.prever(x)).sum::<f64>() / self.arvores.len() as f64
    }
}

fn prever_no(no: &No, x: &[f64]) -> f64 {
    match no {
        No::Folha { valor } => *valor,
        No::Divisao {
            feature,
            limiar,
            esquerda,
            direita,
        } => {
            if x[*feature] <= *limiar {
                prever_no(esquerda, x)
            } else {
                prever_no(direita, x)
            }
        }
    }
}

fn escolher_features(n_features: usize, mtry: usize, rng: &mut XorShift64) -> Vec<usize> {
    let mut features: Vec<usize> = (0..n_features).collect();
    for i in 0..mtry {
        let j = i + rng.usize(n_features - i);
        features.swap(i, j);
    }
    features.truncate(mtry);
    features
}

fn media(amostras: &[AmostraArvore], indices: &[usize]) -> f64 {
    indices.iter().map(|&i| amostras[i].y).sum::<f64>() / indices.len() as f64
}

fn variancia(amostras: &[AmostraArvore], indices: &[usize], media: f64) -> f64 {
    indices
        .iter()
        .map(|&i| {
            let d = amostras[i].y - media;
            d * d
        })
        .sum::<f64>()
        / indices.len() as f64
}

fn melhor_divisao(
    amostras: &[AmostraArvore],
    indices: &[usize],
    features: &[usize],
    min_leaf: usize,
) -> Option<MelhorDivisao> {
    let mut melhor: Option<MelhorDivisao> = None;

    for &feature in features {
        let mut pares: Vec<(f64, f64)> = indices
            .iter()
            .map(|&idx| (amostras[idx].x[feature], amostras[idx].y))
            .collect();
        pares.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let n = pares.len();
        if n < 2 * min_leaf {
            continue;
        }

        let soma_total: f64 = pares.iter().map(|(_, y)| y).sum();
        let soma2_total: f64 = pares.iter().map(|(_, y)| y * y).sum();
        let (mut soma_esq, mut soma2_esq) = (0.0, 0.0);

        for i in 0..(n - 1) {
            let y = pares[i].1;
            soma_esq += y;
            soma2_esq += y * y;

            let n_esq = i + 1;
            let n_dir = n - n_esq;
            if n_esq < min_leaf || n_dir < min_leaf {
                continue;
            }
            if (pares[i].0 - pares[i + 1].0).abs() < 1e-12 {
                continue;
            }

            let soma_dir = soma_total - soma_esq;
            let soma2_dir = soma2_total - soma2_esq;
            let sse_esq = soma2_esq - soma_esq * soma_esq / n_esq as f64;
            let sse_dir = soma2_dir - soma_dir * soma_dir / n_dir as f64;
            let sse = sse_esq + sse_dir;

            let candidato = MelhorDivisao {
                feature,
                limiar: 0.5 * (pares[i].0 + pares[i + 1].0),
                sse,
            };

            if melhor.map(|m| candidato.sse < m.sse).unwrap_or(true) {
                melhor = Some(candidato);
            }
        }
    }

    melhor
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn floresta_aprende_funcao_simples() {
        let amostras: Vec<AmostraArvore> = (0..120)
            .map(|i| {
                let x = i as f64 / 119.0;
                AmostraArvore {
                    x: vec![x],
                    y: if x < 0.5 { 0.2 } else { 0.8 },
                }
            })
            .collect();
        let params = ParametrosFloresta {
            n_arvores: 20,
            max_depth: 4,
            min_samples_split: 8,
            min_samples_leaf: 3,
            mtry: 1,
            seed: 42,
        };
        let floresta = RandomForestRegressor::treinar(&amostras, params);
        assert!(floresta.prever(&[0.2]) < 0.35);
        assert!(floresta.prever(&[0.8]) > 0.65);
    }
}
