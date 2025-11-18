"""
Gerador de Relatório e Visualizações para TCC
Gera gráficos em alta resolução prontos para publicação acadêmica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# Configurar estilo para publicação acadêmica
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18


class ReportGenerator:
    def __init__(self, results_dir):
        """
        Inicializa gerador de relatório.

        Args:
            results_dir: Diretório com results_summary.csv
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'report'
        self.output_dir.mkdir(exist_ok=True)

        # Carregar dados
        self.df = pd.read_csv(self.results_dir / 'results_summary.csv')
        self.df['num_test_samples'] = self.df['test_samples'] // 80  # 80 locutores

        print(f"Dados carregados: {len(self.df)} experimentos")
        print(f"Gráficos serão salvos em: {self.output_dir}")

    def generate_all_plots(self):
        """Gera todos os gráficos para o TCC."""
        print("\n" + "="*80)
        print("GERANDO VISUALIZAÇÕES PARA TCC")
        print("="*80)

        self.plot_accuracy_by_mfcc()
        self.plot_metrics_comparison()
        self.plot_accuracy_distribution()
        self.plot_train_test_comparison()
        self.plot_best_worst_experiments()
        self.plot_correlation_heatmap()
        self.plot_statistical_summary()
        self.generate_latex_tables()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("TODAS AS VISUALIZAÇÕES GERADAS!")
        print(f"Diretório: {self.output_dir}")
        print("="*80)

    def plot_accuracy_by_mfcc(self):
        """Gráfico: Acurácia média por número de MFCCs."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calcular média e desvio padrão
        grouped = self.df.groupby('num_mfccs')['accuracy'].agg(['mean', 'std'])

        # Bar plot com error bars
        x = grouped.index
        y = grouped['mean'] * 100
        yerr = grouped['std'] * 100

        bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.8,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Adicionar valores em cima das barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{y.iloc[i]:.2f}%',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Número de Coeficientes MFCC', fontweight='bold')
        ax.set_ylabel('Acurácia Média (%)', fontweight='bold')
        ax.set_title('Desempenho do Modelo por Configuração de MFCCs',
                    fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_mfcc.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_by_mfcc.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de acurácia por MFCC gerado")

    def plot_metrics_comparison(self):
        """Gráfico: Comparação de todas as métricas."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score']

        for ax, metric, title in zip(axes.flat, metrics, titles):
            grouped = self.df.groupby('num_mfccs')[metric].agg(['mean', 'std'])

            x = grouped.index
            y = grouped['mean'] * 100 if metric == 'accuracy' else grouped['mean']
            yerr = grouped['std'] * 100 if metric == 'accuracy' else grouped['std']

            ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.8)
            ax.set_xlabel('Número de MFCCs', fontweight='bold')
            ylabel = 'Valor (%)' if metric == 'accuracy' else 'Valor'
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Adicionar valores
            for i, (xi, yi) in enumerate(zip(x, y)):
                if metric == 'accuracy':
                    ax.text(xi, yi, f'{yi:.1f}%', ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(xi, yi, f'{yi:.3f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('Comparação de Métricas de Desempenho',
                    fontweight='bold', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'metrics_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de comparação de métricas gerado")

    def plot_accuracy_distribution(self):
        """Gráfico: Distribuição de acurácias (boxplot + violin)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Boxplot
        bp = self.df.boxplot(column='accuracy', by='num_mfccs', ax=ax1,
                             patch_artist=True, return_type='dict')

        # Colorir boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['accuracy']['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_xlabel('Número de MFCCs', fontweight='bold')
        ax1.set_ylabel('Acurácia', fontweight='bold')
        ax1.set_title('Distribuição de Acurácias (Boxplot)', fontweight='bold')
        ax1.get_figure().suptitle('')  # Remove título automático

        # Violin plot
        parts = ax2.violinplot([self.df[self.df['num_mfccs'] == n]['accuracy'].values
                                for n in [10, 20, 30, 40]],
                               positions=[0, 1, 2, 3],
                               showmeans=True, showmedians=True)

        # Colorir violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax2.set_xticks([0, 1, 2, 3])
        ax2.set_xticklabels([10, 20, 30, 40])
        ax2.set_xlabel('Número de MFCCs', fontweight='bold')
        ax2.set_ylabel('Acurácia', fontweight='bold')
        ax2.set_title('Distribuição de Acurácias (Violin Plot)', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_distribution.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_distribution.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de distribuição de acurácias gerado")

    def plot_train_test_comparison(self):
        """Gráfico: Comparação entre experimentos com 1 vs 2 amostras de teste."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Agrupar por MFCCs e número de amostras de teste
        grouped = self.df.groupby(['num_mfccs', 'num_test_samples'])['accuracy'].mean().unstack()

        x = np.arange(len(grouped.index))
        width = 0.35

        bars1 = ax.bar(x - width/2, grouped[1] * 100, width, label='1 amostra de teste',
                      alpha=0.8, color='#2ca02c')
        bars2 = ax.bar(x + width/2, grouped[2] * 100, width, label='2 amostras de teste',
                      alpha=0.8, color='#d62728')

        ax.set_xlabel('Número de MFCCs', fontweight='bold')
        ax.set_ylabel('Acurácia Média (%)', fontweight='bold')
        ax.set_title('Impacto do Número de Amostras de Teste no Desempenho',
                    fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Adicionar valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'train_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'train_test_comparison.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de comparação treino/teste gerado")

    def plot_best_worst_experiments(self):
        """Gráfico: Top 10 melhores e piores experimentos."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Top 10 melhores
        best = self.df.nlargest(10, 'accuracy')
        colors_best = [f"C{i}" for i in range(10)]

        y_pos = np.arange(len(best))
        bars1 = ax1.barh(y_pos, best['accuracy'] * 100, color=colors_best, alpha=0.8)
        ax1.set_yticks(y_pos)
        labels_best = [f"{row['experiment_name'][:20]}\n({row['num_mfccs']} MFCCs)"
                       for _, row in best.iterrows()]
        ax1.set_yticklabels(labels_best, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('Acurácia (%)', fontweight='bold')
        ax1.set_title('Top 10 Melhores Experimentos', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # Adicionar valores
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}%',
                    ha='left', va='center', fontsize=9, fontweight='bold')

        # Top 10 piores
        worst = self.df.nsmallest(10, 'accuracy')
        colors_worst = [f"C{i}" for i in range(10)]

        y_pos = np.arange(len(worst))
        bars2 = ax2.barh(y_pos, worst['accuracy'] * 100, color=colors_worst, alpha=0.8)
        ax2.set_yticks(y_pos)
        labels_worst = [f"{row['experiment_name'][:20]}\n({row['num_mfccs']} MFCCs)"
                        for _, row in worst.iterrows()]
        ax2.set_yticklabels(labels_worst, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Acurácia (%)', fontweight='bold')
        ax2.set_title('Top 10 Piores Experimentos', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        # Adicionar valores
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}%',
                    ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_worst_experiments.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'best_worst_experiments.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de melhores/piores experimentos gerado")

    def plot_correlation_heatmap(self):
        """Gráfico: Heatmap de correlação entre métricas."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Selecionar métricas numéricas
        metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score',
                       'num_mfccs', 'train_samples', 'test_samples']
        corr = self.df[metrics_cols].corr()

        # Heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)

        # Labels em português
        labels = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score',
                 'Num. MFCCs', 'Amostras Treino', 'Amostras Teste']
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        ax.set_title('Matriz de Correlação entre Métricas',
                    fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'correlation_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Heatmap de correlação gerado")

    def plot_statistical_summary(self):
        """Gráfico: Resumo estatístico com intervalo de confiança."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Calcular estatísticas por MFCC
        stats = []
        for mfcc in [10, 20, 30, 40]:
            data = self.df[self.df['num_mfccs'] == mfcc]['accuracy']
            stats.append({
                'mfcc': mfcc,
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q25': data.quantile(0.25),
                'median': data.median(),
                'q75': data.quantile(0.75)
            })

        stats_df = pd.DataFrame(stats)

        x = np.arange(len(stats_df))

        # Plotar média com intervalo de confiança (95%)
        ax.errorbar(x, stats_df['mean'] * 100,
                   yerr=1.96 * stats_df['std'] * 100,  # IC 95%
                   fmt='o-', linewidth=2, markersize=10,
                   capsize=5, capthick=2, label='Média ± IC 95%',
                   color='#1f77b4')

        # Plotar mediana
        ax.plot(x, stats_df['median'] * 100, 's--',
               linewidth=2, markersize=8, label='Mediana',
               color='#ff7f0e')

        # Plotar range (min-max)
        ax.fill_between(x, stats_df['min'] * 100, stats_df['max'] * 100,
                       alpha=0.2, label='Range (min-max)', color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['mfcc'])
        ax.set_xlabel('Número de MFCCs', fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontweight='bold')
        ax.set_title('Resumo Estatístico do Desempenho por Configuração',
                    fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_summary.pdf', bbox_inches='tight')
        plt.close()
        print("✓ Gráfico de resumo estatístico gerado")

    def generate_latex_tables(self):
        """Gera tabelas em formato LaTeX para o TCC."""
        output_file = self.output_dir / 'tables_latex.txt'

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("% Tabelas geradas automaticamente para LaTeX\n\n")

            # Tabela 1: Resumo por MFCCs
            f.write("% Tabela 1: Resumo de Desempenho por Configuração de MFCCs\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Resumo de Desempenho por Configuração de MFCCs}\n")
            f.write("\\label{tab:performance_summary}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{MFCCs} & \\textbf{Acurácia (\\%)} & \\textbf{Precisão} & \\textbf{Revocação} & \\textbf{F1-Score} \\\\\n")
            f.write("\\hline\n")

            for mfcc in [10, 20, 30, 40]:
                data = self.df[self.df['num_mfccs'] == mfcc]
                acc = data['accuracy'].mean() * 100
                prec = data['precision'].mean()
                rec = data['recall'].mean()
                f1 = data['f1_score'].mean()
                f.write(f"{mfcc} & {acc:.2f} $\\pm$ {data['accuracy'].std()*100:.2f} & ")
                f.write(f"{prec:.4f} & {rec:.4f} & {f1:.4f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

            # Tabela 2: Melhor experimento
            f.write("% Tabela 2: Detalhes do Melhor Experimento\n")
            best = self.df.loc[self.df['accuracy'].idxmax()]
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Detalhes do Melhor Experimento}\n")
            f.write("\\label{tab:best_experiment}\n")
            f.write("\\begin{tabular}{|l|l|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Parâmetro} & \\textbf{Valor} \\\\\n")
            f.write("\\hline\n")
            f.write(f"Experimento & {best['experiment_name']} \\\\\n")
            f.write(f"MFCCs & {best['num_mfccs']} \\\\\n")
            f.write(f"Amostras de Treino & {best['train_samples']} \\\\\n")
            f.write(f"Amostras de Teste & {best['test_samples']} \\\\\n")
            f.write(f"Acurácia & {best['accuracy']*100:.2f}\\% \\\\\n")
            f.write(f"Precisão & {best['precision']:.4f} \\\\\n")
            f.write(f"Revocação & {best['recall']:.4f} \\\\\n")
            f.write(f"F1-Score & {best['f1_score']:.4f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"✓ Tabelas LaTeX salvas em: {output_file}")

    def generate_summary_report(self):
        """Gera relatório textual resumido."""
        output_file = self.output_dir / 'summary_report.txt'

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE RESULTADOS - SISTEMA DE RECONHECIMENTO DE LOCUTOR\n")
            f.write("="*80 + "\n\n")

            # Resumo geral
            f.write("1. RESUMO GERAL\n")
            f.write("-"*80 + "\n")
            f.write(f"Total de experimentos: {len(self.df)}\n")
            f.write(f"Acurácia média geral: {self.df['accuracy'].mean()*100:.2f}%\n")
            f.write(f"Desvio padrão: {self.df['accuracy'].std()*100:.2f}%\n")
            f.write(f"Acurácia mínima: {self.df['accuracy'].min()*100:.2f}%\n")
            f.write(f"Acurácia máxima: {self.df['accuracy'].max()*100:.2f}%\n\n")

            # Desempenho por MFCCs
            f.write("2. DESEMPENHO POR CONFIGURAÇÃO DE MFCCs\n")
            f.write("-"*80 + "\n")
            for mfcc in [10, 20, 30, 40]:
                data = self.df[self.df['num_mfccs'] == mfcc]
                f.write(f"\n{mfcc} MFCCs:\n")
                f.write(f"  Acurácia: {data['accuracy'].mean()*100:.2f}% ± {data['accuracy'].std()*100:.2f}%\n")
                f.write(f"  Precisão: {data['precision'].mean():.4f}\n")
                f.write(f"  Revocação: {data['recall'].mean():.4f}\n")
                f.write(f"  F1-Score: {data['f1_score'].mean():.4f}\n")

            # Melhor experimento
            f.write("\n3. MELHOR EXPERIMENTO\n")
            f.write("-"*80 + "\n")
            best = self.df.loc[self.df['accuracy'].idxmax()]
            f.write(f"Experimento: {best['experiment_name']}\n")
            f.write(f"MFCCs: {best['num_mfccs']}\n")
            f.write(f"Divisão treino/teste: {best['train_utterances']} / {best['test_utterances']}\n")
            f.write(f"Acurácia: {best['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score: {best['f1_score']:.4f}\n")

            # Análise treino/teste
            f.write("\n4. ANÁLISE: 1 vs 2 AMOSTRAS DE TESTE\n")
            f.write("-"*80 + "\n")
            for n_test in [1, 2]:
                data = self.df[self.df['num_test_samples'] == n_test]
                f.write(f"\n{n_test} amostra(s) de teste:\n")
                f.write(f"  Acurácia média: {data['accuracy'].mean()*100:.2f}%\n")
                f.write(f"  Número de experimentos: {len(data)}\n")

        print(f"✓ Relatório resumido salvo em: {output_file}")


def main():
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Procurar automaticamente
        from dotenv import load_dotenv
        load_dotenv()
        results_dir = os.path.join(os.getenv('SAVES_PATH', 'saves'), 'experiments')

    if not os.path.exists(os.path.join(results_dir, 'results_summary.csv')):
        print(f"Erro: Arquivo results_summary.csv não encontrado em {results_dir}")
        print("Execute primeiro os experimentos com: python run_experiments.py")
        return

    generator = ReportGenerator(results_dir)
    generator.generate_all_plots()

    print("\n" + "="*80)
    print("GRÁFICOS GERADOS COM SUCESSO!")
    print("="*80)
    print(f"\nDiretório: {generator.output_dir}")
    print("\nArquivos disponíveis:")
    print("  • PNG (para apresentações): *.png")
    print("  • PDF (para TCC/artigos): *.pdf")
    print("  • Tabelas LaTeX: tables_latex.txt")
    print("  • Relatório textual: summary_report.txt")
    print("\nTodos os gráficos estão em alta resolução (300 DPI)")
    print("="*80)


if __name__ == "__main__":
    main()
