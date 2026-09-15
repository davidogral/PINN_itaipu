// neural/forward.rs — Forward pass

// Propaga a entrada pela rede:
//     z = W·a + b ;  a = tanh(z)   (camadas ocultas)
//     a = z                        (camada de saída, linear)
//
// Guarda um `Cache` com z e a de cada camada, necessário ao backprop.
// `tanh` é C∞ — escolha exigida pela PINN (ver docs/decisoes_tecnicas.md §2).

use crate::neural::network::Rede;

/// Valores intermediários do forward, reutilizados no backward.
pub struct Cache {
    /// Ativações: `ativacoes[0]` = entrada; `ativacoes[i+1]` = saída da camada i.
    pub ativacoes: Vec<Vec<f64>>,
    /// Pré-ativações `z` de cada camada (índice alinhado às camadas).
    #[allow(dead_code)] // usado na Fase 5 (derivadas p/ Loss_fisica)
    pub pre_ativacoes: Vec<Vec<f64>>,
}

/// Executa o forward pass e devolve o cache completo.
pub fn forward(rede: &Rede, entrada: &[f64]) -> Cache {
    let l = rede.num_camadas();
    let mut ativacoes: Vec<Vec<f64>> = Vec::with_capacity(l + 1);
    let mut pre_ativacoes: Vec<Vec<f64>> = Vec::with_capacity(l);

    ativacoes.push(entrada.to_vec());

    for (i, camada) in rede.camadas.iter().enumerate() {
        let a_prev = &ativacoes[i];
        let mut z = vec![0.0; camada.vieses.len()];
        for (j, linha) in camada.pesos.iter().enumerate() {
            let mut soma = camada.vieses[j];
            for (k, w) in linha.iter().enumerate() {
                soma += w * a_prev[k];
            }
            z[j] = soma;
        }

        let ultima = i == l - 1;
        let a: Vec<f64> = if ultima {
            z.clone() // saída linear
        } else {
            z.iter().map(|v| v.tanh()).collect() // ocultas: tanh
        };

        pre_ativacoes.push(z);
        ativacoes.push(a);
    }

    Cache {
        ativacoes,
        pre_ativacoes,
    }
}

/// Conveniência: só a saída da rede (descarta o cache).
pub fn prever(rede: &Rede, entrada: &[f64]) -> Vec<f64> {
    forward(rede, entrada).ativacoes.pop().unwrap()
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::neural::network::Rede;

    #[test]
    fn dimensoes_da_saida() {
        let r = Rede::nova(&[1, 16, 16, 1], 1);
        let y = prever(&r, &[0.5]);
        assert_eq!(y.len(), 1);
    }

    #[test]
    fn cache_tem_tamanhos_certos() {
        let r = Rede::nova(&[2, 4, 3], 1);
        let c = forward(&r, &[0.1, -0.2]);
        assert_eq!(c.ativacoes.len(), 3); // entrada + 2 camadas
        assert_eq!(c.pre_ativacoes.len(), 2);
        assert_eq!(c.ativacoes[2].len(), 3); // saída
    }
}
