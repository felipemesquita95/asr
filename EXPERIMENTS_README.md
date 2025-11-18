# Sistema de Experimentos - ASR

Este sistema executa validação cruzada completa com diferentes configurações de MFCCs e divisões treino/teste.

## Experimentos

### Variações de MFCCs
- 10 coeficientes
- 20 coeficientes
- 30 coeficientes
- 40 coeficientes

### Divisões Treino/Teste

#### 1 amostra de teste (5 experimentos - Leave-One-Out)
1. Treino: 1,2,3,4 | Teste: 5
2. Treino: 1,2,3,5 | Teste: 4
3. Treino: 1,2,4,5 | Teste: 3
4. Treino: 1,3,4,5 | Teste: 2
5. Treino: 2,3,4,5 | Teste: 1

#### 2 amostras de teste (10 experimentos)
6. Treino: 1,2,3 | Teste: 4,5
7. Treino: 1,2,4 | Teste: 3,5
8. Treino: 1,3,4 | Teste: 2,5
9. Treino: 2,3,4 | Teste: 1,5
10. Treino: 1,2,5 | Teste: 3,4
11. Treino: 1,3,5 | Teste: 2,4
12. Treino: 2,3,5 | Teste: 1,4
13. Treino: 1,4,5 | Teste: 2,3
14. Treino: 2,4,5 | Teste: 1,3
15. Treino: 3,4,5 | Teste: 1,2

**Total: 4 × 15 = 60 experimentos independentes**

## Como Executar

### Opção 1: Executar tudo do zero
```bash
python run_experiments.py
# Quando perguntado, digite 'n' para NÃO pular pré-processamento
```

### Opção 2: Pular pré-processamento (se já foi feito)
```bash
python run_experiments.py
# Quando perguntado, digite 's' para pular pré-processamento
```

## Estrutura de Resultados

```
saves/experiments/
├── mfcc10/
│   ├── exp01_train1234_test5/
│   │   ├── model.keras
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── result.json
│   ├── exp02_train1235_test4/
│   │   └── ...
│   └── ...
├── mfcc20/
│   └── ...
├── mfcc30/
│   └── ...
├── mfcc40/
│   └── ...
├── results_summary.json    # Todos os resultados em JSON
└── results_summary.csv     # Todos os resultados em CSV
```

## Resultados

Ao final, o sistema gera:

1. **results_summary.json**: Todos os resultados detalhados
2. **results_summary.csv**: Planilha para análise no Excel/LibreOffice
3. **Relatórios no terminal**:
   - Resumo por número de MFCCs
   - Resumo por tipo de experimento (1 vs 2 amostras de teste)
   - Melhor experimento

## Métricas Coletadas

Para cada experimento:
- Acurácia
- Precisão (macro-average)
- Recall (macro-average)
- F1-score (macro-average)
- Matriz de confusão
- Curvas de treinamento (loss e accuracy)

## Tempo Estimado

- Pré-processamento: ~30-60 min (depende do hardware)
- Cada experimento: ~10-30 min (depende do hardware e GPU)
- **Total**: ~10-30 horas (com GPU pode ser mais rápido)

## Isolamento de Experimentos

✅ Cada experimento:
- Treina modelo **independente**
- Usa dados **separados** (sem contaminação)
- Salva em diretório **próprio**
- Não compartilha pesos entre modelos

## Análise de Resultados

Após executar, você pode:

1. Abrir `results_summary.csv` no Excel
2. Criar gráficos comparando:
   - Acurácia por número de MFCCs
   - Acurácia por número de amostras de teste
   - F1-score por configuração
3. Identificar melhor configuração

## Exemplo de Análise

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv('saves/experiments/results_summary.csv')

# Média de acurácia por MFCCs
print(df.groupby('num_mfccs')['accuracy'].mean())

# Melhor experimento
best = df.loc[df['accuracy'].idxmax()]
print(f"Melhor: {best['experiment_name']} com {best['accuracy']*100:.2f}%")
```

## Customização

Para modificar experimentos, edite `run_experiments.py`:

```python
# Adicionar mais MFCCs
NUM_MFCC_OPTIONS = [10, 20, 30, 40, 50]

# Adicionar experimentos customizados
EXPERIMENTS_CUSTOM = [
    {"train": [1, 2], "test": [3, 4, 5], "name": "custom_exp"},
]
```
