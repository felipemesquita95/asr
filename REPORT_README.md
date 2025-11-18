# Gerador de Relatório e Visualizações para TCC

Script automático que gera **gráficos em alta resolução** prontos para uso em TCC, apresentações e artigos científicos.

## Como Usar

### Após executar os experimentos:

```bash
python generate_report.py
```

Ou especificando o diretório:

```bash
python generate_report.py /caminho/para/saves/experiments
```

## Gráficos Gerados

### 1. **Acurácia por MFCCs** (`accuracy_by_mfcc.png/pdf`)
- Bar chart com média e desvio padrão
- Mostra qual configuração de MFCCs tem melhor desempenho
- **Uso**: Comparação de configurações

### 2. **Comparação de Métricas** (`metrics_comparison.png/pdf`)
- 4 subgráficos: Acurácia, Precisão, Revocação, F1-Score
- Todos por número de MFCCs
- **Uso**: Análise completa de desempenho

### 3. **Distribuição de Acurácias** (`accuracy_distribution.png/pdf`)
- Boxplot + Violin plot
- Mostra variabilidade e outliers
- **Uso**: Análise estatística da robustez

### 4. **Comparação Treino/Teste** (`train_test_comparison.png/pdf`)
- Compara 1 vs 2 amostras de teste
- Mostra impacto do tamanho do conjunto de teste
- **Uso**: Justificar escolha de divisão treino/teste

### 5. **Melhores/Piores Experimentos** (`best_worst_experiments.png/pdf`)
- Top 10 de cada
- Identifica padrões de sucesso/falha
- **Uso**: Análise de resultados extremos

### 6. **Heatmap de Correlação** (`correlation_heatmap.png/pdf`)
- Correlação entre todas as métricas
- Identifica relações entre variáveis
- **Uso**: Análise multivariada

### 7. **Resumo Estatístico** (`statistical_summary.png/pdf`)
- Média com intervalo de confiança 95%
- Mediana e range
- **Uso**: Apresentação de resultados com significância estatística

## Outros Arquivos Gerados

### **Tabelas LaTeX** (`tables_latex.txt`)
Tabelas prontas para copiar/colar no seu TCC:
- Tabela resumo por MFCCs
- Tabela do melhor experimento
- Formatação IEEE/ACM

**Exemplo de uso no LaTeX:**
```latex
\input{report/tables_latex.txt}
```

### **Relatório Textual** (`summary_report.txt`)
Resumo em texto com:
- Estatísticas gerais
- Desempenho por configuração
- Melhor experimento
- Análise comparativa

## Formatos Disponíveis

Cada gráfico é gerado em **2 formatos**:

### PNG (300 DPI)
- Para apresentações PowerPoint/Google Slides
- Para visualização rápida
- Tamanho otimizado

### PDF (vetorial)
- Para TCC/dissertação/artigos
- Qualidade infinita (vetorial)
- Aceito por revistas científicas

## Estrutura de Saída

```
saves/experiments/report/
├── accuracy_by_mfcc.png
├── accuracy_by_mfcc.pdf
├── metrics_comparison.png
├── metrics_comparison.pdf
├── accuracy_distribution.png
├── accuracy_distribution.pdf
├── train_test_comparison.png
├── train_test_comparison.pdf
├── best_worst_experiments.png
├── best_worst_experiments.pdf
├── correlation_heatmap.png
├── correlation_heatmap.pdf
├── statistical_summary.png
├── statistical_summary.pdf
├── tables_latex.txt
└── summary_report.txt
```

## Customização

Para modificar os gráficos, edite `generate_report.py`:

```python
# Alterar cores
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#FF0000', '#00FF00'])

# Alterar tamanho de fonte
plt.rcParams['font.size'] = 14

# Alterar DPI (resolução)
plt.savefig('output.png', dpi=600)  # 600 DPI para impressão

# Alterar estilo
plt.style.use('ggplot')  # ou 'seaborn', 'bmh', etc.
```

## Dicas para TCC

### Para a Apresentação
Use os arquivos `.png` - são leves e carregam rápido no PowerPoint/Google Slides.

### Para o Documento
Use os arquivos `.pdf`:
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{report/accuracy_by_mfcc.pdf}
  \caption{Desempenho do modelo por configuração de MFCCs.}
  \label{fig:accuracy_mfcc}
\end{figure}
```

### Tabelas LaTeX
Copie direto de `tables_latex.txt` para seu documento:
```latex
\input{report/tables_latex.txt}
```

### Análise Textual
Use `summary_report.txt` como base para escrever a seção de Resultados.

## Requisitos

Já incluído em `requirements.txt`:
- pandas
- numpy
- matplotlib
- seaborn

## Troubleshooting

**Erro: "results_summary.csv não encontrado"**
- Execute primeiro: `python run_experiments.py`

**Gráficos não aparecem**
- Normal! Eles são salvos em arquivos, não mostrados na tela
- Verifique o diretório: `saves/experiments/report/`

**Fontes pequenas demais**
- Edite `plt.rcParams['font.size']` no início do script

**Cores não agradam**
- Edite `sns.set_palette()` ou use temas diferentes
