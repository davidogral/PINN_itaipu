// neural/network.rs — Estrutura da rede neural

// Rede feedforward genérica. Arquitetura-alvo do projeto:
//     1 → [16, tanh] → [16, tanh] → 1 (linear)
//
// Inclui um PRNG próprio (xorshift64*) porque o projeto não usa libs
// externas de RNG — pesos inicializados por Xavier/Glorot uniforme.

/// Gerador pseudoaleatório determinístico (xorshift64*), para init reprodutível.
pub struct Prng {
    estado: u64,
}

impl Prng {
    pub fn nova(semente: u64) -> Self {
        // estado não-nulo
        Prng {
            estado: semente ^ 0x9E3779B97F4A7C15 | 1,
        }
    }

    fn proximo_u64(&mut self) -> u64 {
        let mut x = self.estado;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.estado = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Float uniforme em [0, 1).
    pub fn uniforme(&mut self) -> f64 {
        (self.proximo_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Float uniforme em [lo, hi).
    pub fn uniforme_em(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniforme()
    }
}

/// Uma camada densa: `saida = ativacao(W · entrada + b)`.
#[derive(Clone)]
pub struct Camada {
    pub pesos: Vec<Vec<f64>>, // [n_saida][n_entrada]
    pub vieses: Vec<f64>,     // [n_saida]
}

/// Rede feedforward (multilayer perceptron).
#[derive(Clone)]
pub struct Rede {
    #[allow(dead_code)] // metadado usado em logs/serialização na Fase 5
    pub arquitetura: Vec<usize>,
    pub camadas: Vec<Camada>,
}

impl Rede {
    /// Cria a rede a partir da arquitetura (ex.: `&[1, 16, 16, 1]`).
    ///
    /// Pesos: Xavier/Glorot uniforme em `±sqrt(6/(fan_in+fan_out))`.
    /// Vieses: zero. `semente` torna a inicialização reprodutível.
    pub fn nova(arquitetura: &[usize], semente: u64) -> Self {
        assert!(arquitetura.len() >= 2, "rede precisa de >= 2 camadas");
        let mut prng = Prng::nova(semente);
        let mut camadas = Vec::with_capacity(arquitetura.len() - 1);

        for par in arquitetura.windows(2) {
            let (fan_in, fan_out) = (par[0], par[1]);
            let limite = (6.0 / (fan_in + fan_out) as f64).sqrt();
            let pesos = (0..fan_out)
                .map(|_| {
                    (0..fan_in)
                        .map(|_| prng.uniforme_em(-limite, limite))
                        .collect()
                })
                .collect();
            let vieses = vec![0.0; fan_out];
            camadas.push(Camada { pesos, vieses });
        }

        Rede {
            arquitetura: arquitetura.to_vec(),
            camadas,
        }
    }

    /// Número de camadas com pesos (= arquitetura.len() - 1).
    pub fn num_camadas(&self) -> usize {
        self.camadas.len()
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn arquitetura_correta() {
        let r = Rede::nova(&[1, 16, 16, 1], 42);
        assert_eq!(r.num_camadas(), 3);
        assert_eq!(r.camadas[0].pesos.len(), 16); // fan_out
        assert_eq!(r.camadas[0].pesos[0].len(), 1); // fan_in
        assert_eq!(r.camadas[2].pesos.len(), 1);
        assert_eq!(r.camadas[2].pesos[0].len(), 16);
    }

    #[test]
    fn init_reprodutivel_e_no_intervalo() {
        let a = Rede::nova(&[2, 8, 1], 7);
        let b = Rede::nova(&[2, 8, 1], 7);
        assert_eq!(a.camadas[0].pesos[0][0], b.camadas[0].pesos[0][0]);
        // Xavier limite p/ camada 2->8: sqrt(6/10) ≈ 0.7746
        let lim = (6.0_f64 / 10.0).sqrt();
        for linha in &a.camadas[0].pesos {
            for w in linha {
                assert!(w.abs() <= lim, "peso {w} fora de ±{lim}");
            }
        }
    }
}
