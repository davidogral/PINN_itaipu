// math/newton_raphson.rs — Método de Newton-Raphson (do zero)

// Encontra raízes de uma função 1D pela iteração de Newton:
//     x_{n+1} = x_n - f(x_n) / f'(x_n)
//
// Para achar EXTREMOS (ponto ótimo de geração na Fase 5), aplica-se
// Newton sobre a derivada: a raiz de f'(x)=0 é o extremo, usando f''(x).
// O helper `extremo` faz exatamente isso.

/// Parâmetros de parada da iteração.
pub struct Config {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            tol: 1e-10,
            max_iter: 100,
        }
    }
}

/// Encontra uma raiz de `f` partindo de `x0`, dada a derivada `df`.
///
/// Retorna `None` se a derivada zerar (divisão por ~0) ou se não convergir
/// dentro de `max_iter`.
pub fn raiz<F, DF>(f: F, df: DF, x0: f64, cfg: &Config) -> Option<f64>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;
    for _ in 0..cfg.max_iter {
        let fx = f(x);
        if fx.abs() < cfg.tol {
            return Some(x);
        }
        let dfx = df(x);
        if dfx.abs() < 1e-14 {
            return None; // derivada ~ 0: passo indefinido
        }
        let proximo = x - fx / dfx;
        if (proximo - x).abs() < cfg.tol {
            return Some(proximo);
        }
        x = proximo;
    }
    None
}

/// Encontra um extremo (mín./máx.) de `g` partindo de `x0`, dadas a 1ª
/// derivada `dg` e a 2ª derivada `d2g`. Equivale a achar a raiz de `dg`.
#[allow(dead_code)]
pub fn extremo<DG, D2G>(dg: DG, d2g: D2G, x0: f64, cfg: &Config) -> Option<f64>
where
    DG: Fn(f64) -> f64,
    D2G: Fn(f64) -> f64,
{
    raiz(dg, d2g, x0, cfg)
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn raiz_de_x2_menos_2() {
        // f(x) = x² - 2  => raiz = sqrt(2)
        let cfg = Config::default();
        let r = raiz(|x| x * x - 2.0, |x| 2.0 * x, 1.0, &cfg).unwrap();
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn extremo_de_parabola() {
        // g(x) = -(x-3)²  => máximo em x=3
        // g'(x) = -2(x-3) ; g''(x) = -2
        let cfg = Config::default();
        let x = extremo(|x| -2.0 * (x - 3.0), |_| -2.0, 0.0, &cfg).unwrap();
        assert!((x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn derivada_nula_retorna_none() {
        // f(x) = x² + 1 (sem raiz real) com x0=0 => f(0)=1≠0 e f'(0)=0
        // => passo indefinido => None. (Não usar x²: x=0 já é raiz dupla.)
        let cfg = Config::default();
        assert!(raiz(|x| x * x + 1.0, |x| 2.0 * x, 0.0, &cfg).is_none());
    }
}
