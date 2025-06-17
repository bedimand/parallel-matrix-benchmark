#!/usr/bin/env python3
"""
Teste de diferentes tamanhos de matriz para comparar CPU vs GPU
"""

from matrix_comparison import CompleteMatrixBenchmark
import matplotlib.pyplot as plt
import numpy as np

def test_different_sizes():
    print('🧪 TESTE COMPARATIVO COM DIFERENTES TAMANHOS')
    print('='*60)
    
    sizes = [1000, 2000, 3000, 4000]
    results = []
    
    for size in sizes:
        print(f'\n📐 Testando matriz {size}x{size} ({size**2 * 4 / (1024**2):.1f} MB por matriz)')
        
        try:
            benchmark = CompleteMatrixBenchmark(matrix_size=size)
            matrix_a, matrix_b = benchmark.generate_matrices()
            
            # CPU sequencial
            _, cpu_time = benchmark.cpu_sequential(matrix_a, matrix_b)
            
            # CPU paralelo (8 threads - melhor resultado anterior)
            _, cpu_par_time = benchmark.cpu_parallel(matrix_a, matrix_b, 8)
            cpu_speedup = cpu_time / cpu_par_time
            
            # GPU
            gpu_result = None
            if benchmark.gpu_available:
                try:
                    _, gpu_total, gpu_compute = benchmark.gpu_cupy(matrix_a, matrix_b)
                    speedup_total = cpu_time / gpu_total
                    speedup_compute = cpu_time / gpu_compute
                    
                    print(f'  CPU Sequencial: {cpu_time:.4f}s')
                    print(f'  CPU 8 threads:  {cpu_par_time:.4f}s (speedup: {cpu_speedup:.2f}x)')
                    print(f'  GPU Total:      {gpu_total:.4f}s (speedup: {speedup_total:.2f}x)')
                    print(f'  GPU Computação: {gpu_compute:.4f}s (speedup: {speedup_compute:.2f}x)')
                    
                    # Compara melhor CPU vs melhor GPU
                    best_cpu = min(cpu_time, cpu_par_time)
                    advantage = best_cpu / gpu_compute
                    if advantage > 1:
                        print(f'  🚀 GPU {advantage:.2f}x mais rápida que melhor CPU!')
                    else:
                        print(f'  💻 CPU {1/advantage:.2f}x mais rápida que GPU')
                        
                    results.append({
                        'size': size,
                        'cpu_seq': cpu_time,
                        'cpu_par': cpu_par_time,
                        'gpu_total': gpu_total,
                        'gpu_compute': gpu_compute,
                        'gpu_advantage': advantage
                    })
                        
                except Exception as e:
                    print(f'  ❌ Erro GPU: {e}')
            else:
                print(f'  CPU Sequencial: {cpu_time:.4f}s')
                print(f'  CPU 8 threads:  {cpu_par_time:.4f}s (speedup: {cpu_speedup:.2f}x)')
                print('  GPU: Não disponível')
                
        except Exception as e:
            print(f'  ❌ Erro no teste: {e}')
    
    # Resumo final
    if results:
        print('\n' + '='*60)
        print('📊 RESUMO DOS RESULTADOS')
        print('='*60)
        print(f"{'Tamanho':<8} {'CPU Seq':<10} {'CPU Par':<10} {'GPU Comp':<10} {'Vantagem GPU':<12}")
        print('-' * 60)
        
        for r in results:
            print(f"{r['size']:<8} {r['cpu_seq']:<10.4f} {r['cpu_par']:<10.4f} {r['gpu_compute']:<10.4f} {r['gpu_advantage']:<12.2f}x")
        
        # Encontra ponto de break-even
        gpu_wins = [r for r in results if r['gpu_advantage'] > 1]
        if gpu_wins:
            smallest_win = min(gpu_wins, key=lambda x: x['size'])
            print(f'\n🎯 GPU começa a vencer a partir de {smallest_win["size"]}x{smallest_win["size"]}')
        else:
            print('\n💻 CPU foi mais rápida em todos os tamanhos testados')
        
        # Cria visualizações
        create_scalability_plots(results)

def create_scalability_plots(results):
    """Cria gráficos simples de comparação de performance"""
    if not results:
        print("Sem dados para plotar")
        return
    
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extrai dados
    sizes = [r['size'] for r in results]
    cpu_par_times = [r['cpu_par'] for r in results]
    gpu_compute_times = [r['gpu_compute'] for r in results]
    gpu_advantages = [r['gpu_advantage'] for r in results]
    
    # 1. Comparação direta de tempo
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_par_times, width, label='CPU (8 threads)', 
                    color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, gpu_compute_times, width, label='GPU (CUDA)', 
                    color='red', alpha=0.7)
    
    ax1.set_xlabel('Tamanho da Matriz')
    ax1.set_ylabel('Tempo de Execução (s)')
    ax1.set_title('Comparação de Performance: CPU vs GPU')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{size}x{size}' for size in sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(cpu_par_times + gpu_compute_times)*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # 2. Vantagem da GPU
    bars = ax2.bar(range(len(sizes)), gpu_advantages, 
                   color=['red' if adv < 1 else 'green' for adv in gpu_advantages], 
                   alpha=0.7)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')
    ax2.set_xlabel('Tamanho da Matriz')
    ax2.set_ylabel('Vantagem da GPU (x)')
    ax2.set_title('GPU vs CPU: Fator de Melhoria')
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{size}x{size}' for size in sizes])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adiciona valores e anotações
    for i, (bar, adv) in enumerate(zip(bars, gpu_advantages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(gpu_advantages)*0.02,
                f'{adv:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Adiciona indicação de vencedor
        if adv > 1:
            ax2.text(bar.get_x() + bar.get_width()/2., height/2, '🚀\nGPU', 
                    ha='center', va='center', fontweight='bold', fontsize=10, color='white')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height/2, '💻\nCPU', 
                    ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
    # Adiciona sumário
    gpu_wins = len([adv for adv in gpu_advantages if adv > 1])
    fig.suptitle(f'Análise de Escalabilidade: CPU vs GPU RTX 3070 Ti\n'
                f'GPU vence em {gpu_wins}/{len(sizes)} tamanhos testados', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Gráfico salvo como: scalability_analysis.png")

if __name__ == "__main__":
    test_different_sizes() 