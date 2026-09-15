// math/interpolacao.rs — Interpolação Quadrática (do zero)

// Ajusta uma parábola por 3 pontos para estimar valores intermediários
// da relação vazão × geração. Implementado sem bibliotecas.

// Três formas, todas equivalentes numericamente:
//   - lagrange_quadratica:        forma de Lagrange
//   - newton_quadratica:          diferenças divididas de Newton
//   - coeficientes_quadratica:    devolve (a, b, c) de a·x² + b·x + c

// parábola dá o extremo analítico, e a derivada 2·a·x + b alimenta o
// Newton-Raphson.

/// Interpolação quadrática pela forma de Lagrange.
///
/// `p` são 3 pontos `(x, y)` com `x` distintos. Avalia o polinômio em `x`.
pub fn lagrange_quadratica(p: [(f64, f64); 3], x: f64) -> f64 {
    let (x0, y0) = p[0];
    let (x1, y1) = p[1];
    let (x2, y2) = p[2];

    let l0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2));
    let l1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2));
    let l2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1));

    y0 * l0 + y1 * l1 + y2 * l2
}

/// Interpolação quadrática por diferenças divididas de Newton.
pub fn newton_quadratica(p: [(f64, f64); 3], x: f64) -> f64 {
    let (x0, y0) = p[0];
    let (x1, y1) = p[1];
    let (x2, y2) = p[2];

    let f01 = (y1 - y0) / (x1 - x0);
    let f12 = (y2 - y1) / (x2 - x1);
    let f012 = (f12 - f01) / (x2 - x0);

    // b0 + b1·(x-x0) + b2·(x-x0)·(x-x1)
    y0 + f01 * (x - x0) + f012 * (x - x0) * (x - x1)
}

/// Coeficientes `(a, b, c)` da parábola `a·x² + b·x + c` que passa pelos 3 pontos.
///
/// Deriva da forma de Newton expandida; útil para achar o vértice/derivada.
pub fn coeficientes_quadratica(p: [(f64, f64); 3]) -> (f64, f64, f64) {
    let (x0, y0) = p[0];
    let (x1, y1) = p[1];
    let (x2, y2) = p[2];

    let f01 = (y1 - y0) / (x1 - x0);
    let f12 = (y2 - y1) / (x2 - x1);
    let a = (f12 - f01) / (x2 - x0); // termo de 2ª ordem
    let b = f01 - a * (x0 + x1);
    let c = y0 - f01 * x0 + a * x0 * x1;
    (a, b, c)
}

/// Avalia `a·x² + b·x + c`.
#[allow(dead_code)] // usado na Fase 5 (otimização)
pub fn avaliar_quadratica(coef: (f64, f64, f64), x: f64) -> f64 {
    let (a, b, c) = coef;
    a * x * x + b * x + c
}

/// Vértice (x do extremo) da parábola: `x* = -b / (2a)`.
/// Retorna `None` se `a ≈ 0` (reta, sem extremo).
pub fn vertice_quadratica(coef: (f64, f64, f64)) -> Option<f64> {
    let (a, b, _) = coef;
    if a.abs() < 1e-12 {
        None
    } else {
        Some(-b / (2.0 * a))
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    const PONTOS: [(f64, f64); 3] = [(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]; // y = x²

    #[test]
    fn lagrange_reproduz_quadratica() {
        // y = x² => em x=2.5 deve dar 6.25
        assert!((lagrange_quadratica(PONTOS, 2.5) - 6.25).abs() < 1e-9);
    }

    #[test]
    fn lagrange_e_newton_concordam() {
        for &x in &[0.5, 1.7, 2.3, 4.0] {
            let a = lagrange_quadratica(PONTOS, x);
            let b = newton_quadratica(PONTOS, x);
            assert!((a - b).abs() < 1e-9, "x={x}: {a} vs {b}");
        }
    }

    #[test]
    fn coeficientes_de_x_quadrado() {
        let (a, b, c) = coeficientes_quadratica(PONTOS);
        assert!((a - 1.0).abs() < 1e-9);
        assert!(b.abs() < 1e-9);
        assert!(c.abs() < 1e-9);
    }

    #[test]
    fn vertice_de_parabola_concava() {
        // y = -(x-3)² + 5 = -x² + 6x - 4  => vértice em x=3
        let pts = [(1.0, 1.0), (3.0, 5.0), (5.0, 1.0)];
        let coef = coeficientes_quadratica(pts);
        let v = vertice_quadratica(coef).unwrap();
        assert!((v - 3.0).abs() < 1e-9);
    }
}
