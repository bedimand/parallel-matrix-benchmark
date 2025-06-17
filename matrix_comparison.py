#!/usr/bin/env python3
"""
Comparação Completa - CPU vs GPU
Trabalho de Aceleração em Ciência de Dados usando Computação Paralela

Este programa executa uma comparação completa entre processamento paralelo
na CPU (1, 2, 4, 8, todas threads) vs GPU CUDA.

Autor: Trabalho de Faculdade
Data: 2024
"""

import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import os
from concurrent.futures import ProcessPoolExecutor

# Imports condicionais para GPU
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

def _multiply_chunk_global(args):
    """Função global para multiplicação de chunk (necessária para multiprocessing)"""
    chunk_a, matrix_b = args
    return np.dot(chunk_a, matrix_b)

class CompleteMatrixBenchmark:
    def __init__(self, matrix_size=2000):
        """
        Inicializa o benchmark completo
        
        Args:
            matrix_size (int): Tamanho das matrizes quadradas
        """
        self.matrix_size = matrix_size
        self.max_threads = mp.cpu_count()
        self.gpu_available = self.check_gpu()
        
        print("="*80)
        print("BENCHMARK COMPLETO - PROCESSAMENTO PARALELO DE MATRIZES")
        print("="*80)
        print(f"Sistema: {psutil.virtual_memory().total // (1024**3)} GB RAM")
        print(f"CPU: {self.max_threads} cores/threads")
        print(f"Matriz: {matrix_size}x{matrix_size} ({matrix_size**2 * 4 / (1024**2):.1f} MB por matriz)")
        
        if self.gpu_available:
            print("GPU: Disponível")
        else:
            print("GPU: Não disponível")
        print("="*80)
    
    def check_gpu(self):
        """Verifica se GPU está disponível"""
        if CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.getDeviceCount()
                device = cp.cuda.Device(0)
                print(f"GPU detectada: RTX 3070 Ti (Device {device.id})")
                return True
            except Exception as e:
                print(f"Erro ao detectar GPU: {e}")
                pass
        return False
    
    def generate_matrices(self):
        """Gera matrizes de teste"""
        np.random.seed(42)
        matrix_a = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        matrix_b = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        return matrix_a, matrix_b
    
    def cpu_sequential(self, matrix_a, matrix_b):
        """Execução sequencial (1 thread)"""
        print("🔄 Executando CPU sequencial (1 thread)...")
        
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        
        start_time = time.time()
        result = np.dot(matrix_a, matrix_b)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Tempo: {execution_time:.4f}s")
        return result, execution_time
    
    def cpu_parallel(self, matrix_a, matrix_b, num_threads):
        """Execução paralela CPU com N threads"""
        print(f"🔄 Executando CPU paralelo ({num_threads} threads)...")
        
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        
        start_time = time.time()
        result = np.dot(matrix_a, matrix_b)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Tempo: {execution_time:.4f}s")
        return result, execution_time
    
    def cpu_manual_parallel(self, matrix_a, matrix_b, num_processes):
        """Paralelização manual usando multiprocessing"""
        print(f"🔄 Executando paralelização manual ({num_processes} processos)...")
        
        start_time = time.time()
        
        # Divide a matriz em chunks
        chunk_size = self.matrix_size // num_processes
        chunks = []
        
        for i in range(num_processes):
            start_row = i * chunk_size
            end_row = (i + 1) * chunk_size if i < num_processes - 1 else self.matrix_size
            chunk_a = matrix_a[start_row:end_row, :]
            chunks.append((chunk_a, matrix_b))
        
        # Executa em paralelo
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_multiply_chunk_global, chunks))
        
        # Combina resultados
        result = np.vstack(results)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Tempo: {execution_time:.4f}s")
        return result, execution_time
    
    def gpu_cupy(self, matrix_a, matrix_b):
        """Execução GPU usando CuPy"""
        if not CUPY_AVAILABLE:
            return None, float('inf'), float('inf')
            
        print("🚀 Executando GPU (CuPy)...")
        
        # Transferência para GPU
        start_transfer = time.time()
        gpu_a = cp.asarray(matrix_a)
        gpu_b = cp.asarray(matrix_b)
        transfer_time = time.time() - start_transfer
        
        # Computação GPU
        start_compute = time.time()
        gpu_result = cp.dot(gpu_a, gpu_b)
        cp.cuda.Stream.null.synchronize()
        compute_time = time.time() - start_compute
        
        # Transferência de volta
        start_back = time.time()
        result = cp.asnumpy(gpu_result)
        back_time = time.time() - start_back
        
        total_time = transfer_time + compute_time + back_time
        
        print(f"   ✅ Tempo total: {total_time:.4f}s")
        print(f"      - Transferência: {transfer_time:.4f}s")
        print(f"      - Computação: {compute_time:.4f}s")
        print(f"      - Volta: {back_time:.4f}s")
        
        return result, total_time, compute_time
    
    def run_complete_benchmark(self):
        """Executa benchmark completo"""
        print("\n📊 INICIANDO BENCHMARK COMPLETO...")
        
        # Gera matrizes
        matrix_a, matrix_b = self.generate_matrices()
        
        results = {
            'method': [],
            'threads': [],
            'time': [],
            'speedup': [],
            'efficiency': [],
            'type': []
        }
        
        # 1. CPU Sequencial (baseline)
        cpu_result, seq_time = self.cpu_sequential(matrix_a, matrix_b)
        baseline_time = seq_time
        
        results['method'].append('CPU Sequencial')
        results['threads'].append(1)
        results['time'].append(seq_time)
        results['speedup'].append(1.0)
        results['efficiency'].append(1.0)
        results['type'].append('CPU')
        
        # 2. CPU Paralelo (2, 4, 8, todas threads)
        thread_configs = [2, 4, 8, self.max_threads]
        
        for threads in thread_configs:
            if threads <= self.max_threads:
                # NumPy paralelo
                _, par_time = self.cpu_parallel(matrix_a, matrix_b, threads)
                speedup = baseline_time / par_time
                efficiency = speedup / threads
                
                results['method'].append(f'CPU Paralelo (NumPy)')
                results['threads'].append(threads)
                results['time'].append(par_time)
                results['speedup'].append(speedup)
                results['efficiency'].append(efficiency)
                results['type'].append('CPU')
                
                # Paralelo manual
                if threads <= 8:  # Evita overhead excessivo
                    _, manual_time = self.cpu_manual_parallel(matrix_a, matrix_b, threads)
                    speedup_manual = baseline_time / manual_time
                    efficiency_manual = speedup_manual / threads
                    
                    results['method'].append(f'CPU Manual')
                    results['threads'].append(threads)
                    results['time'].append(manual_time)
                    results['speedup'].append(speedup_manual)
                    results['efficiency'].append(efficiency_manual)
                    results['type'].append('CPU')
        
        # 3. GPU
        if self.gpu_available:
            gpu_result, gpu_time, gpu_compute = self.gpu_cupy(matrix_a, matrix_b)
            
            if gpu_result is not None:
                # GPU Total
                speedup_total = baseline_time / gpu_time
                results['method'].append('GPU Total')
                results['threads'].append('GPU')
                results['time'].append(gpu_time)
                results['speedup'].append(speedup_total)
                results['efficiency'].append(speedup_total)  # Para GPU, efficiency = speedup
                results['type'].append('GPU')
                
                # GPU Apenas Computação
                speedup_compute = baseline_time / gpu_compute
                results['method'].append('GPU Computação')
                results['threads'].append('GPU')
                results['time'].append(gpu_compute)
                results['speedup'].append(speedup_compute)
                results['efficiency'].append(speedup_compute)
                results['type'].append('GPU')
                
                # Verifica consistência
                if np.allclose(cpu_result, gpu_result, rtol=1e-5):
                    print("✅ Resultados CPU e GPU são consistentes")
                else:
                    print("⚠️  Diferença detectada entre CPU e GPU")
        
        return results
    
    def create_comprehensive_plots(self, results):
        """Cria visualizações focadas e claras"""
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepara dados
        df = pd.DataFrame(results)
        cpu_data = df[df['type'] == 'CPU'].copy()
        gpu_data = df[df['type'] == 'GPU'].copy()
        
        # 1. Tempo de execução por método
        methods = df['method'].tolist()
        times = df['time'].tolist()
        
        # Cores diferentes para CPU vs GPU
        colors = []
        for method in methods:
            if 'GPU' in method:
                colors.append('red')
            elif 'Manual' in method:
                colors.append('orange')
            else:
                colors.append('blue')
        
        bars = ax1.bar(range(len(methods)), times, color=colors, alpha=0.7)
        ax1.set_ylabel('Tempo de Execução (s)')
        ax1.set_title(f'Tempo de Execução - Matriz {self.matrix_size}x{self.matrix_size}')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Adiciona valores nas barras
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                    f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='CPU NumPy'),
            Patch(facecolor='orange', alpha=0.7, label='CPU Manual'),
            Patch(facecolor='red', alpha=0.7, label='GPU CUDA')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. Comparação CPU vs GPU
        if not gpu_data.empty:
            categories = ['Melhor CPU', 'GPU (Total)', 'GPU (Só Computação)']
            
            # Encontra melhor CPU
            cpu_best_time = cpu_data['time'].min()
            cpu_best_speedup = df[df['method'] == 'CPU Sequencial']['time'].iloc[0] / cpu_best_time
            
            # GPU resultados
            gpu_total = gpu_data[gpu_data['method'] == 'GPU Total']['speedup'].iloc[0]
            gpu_compute = gpu_data[gpu_data['method'] == 'GPU Computação']['speedup'].iloc[0]
            
            values = [cpu_best_speedup, gpu_total, gpu_compute]
            colors_comp = ['blue', 'red', 'darkred']
            
            bars = ax2.bar(categories, values, color=colors_comp, alpha=0.7)
            ax2.set_ylabel('Speedup vs CPU Sequencial')
            ax2.set_title('Comparação Final: CPU vs GPU')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
            
            # Adiciona valores
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                        f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Adiciona análise
            winner = "GPU" if gpu_compute > cpu_best_speedup else "CPU"
            if winner == "GPU":
                advantage = gpu_compute / cpu_best_speedup
                ax2.text(0.5, 0.95, f'🏆 GPU vence por {advantage:.1f}x', 
                        transform=ax2.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
                        fontsize=12, fontweight='bold')
            else:
                advantage = cpu_best_speedup / gpu_compute
                ax2.text(0.5, 0.95, f'🏆 CPU vence por {advantage:.1f}x', 
                        transform=ax2.transAxes, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                        fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'GPU não disponível', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16)
            ax2.set_title('GPU não detectada')
        
        plt.tight_layout()
        plt.savefig('complete_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_complete_results(self, results):
        """Salva resultados completos"""
        df = pd.DataFrame(results)
        df.to_csv('complete_benchmark_results.csv', index=False)
        
        print("\n" + "="*80)
        print("RESUMO COMPLETO DOS RESULTADOS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Análise detalhada
        print("\n" + "="*80)
        print("ANÁLISE DETALHADA")
        print("="*80)
        
        cpu_data = df[df['type'] == 'CPU']
        gpu_data = df[df['type'] == 'GPU']
        
        if not cpu_data.empty:
            best_cpu = cpu_data.loc[cpu_data['speedup'].idxmax()]
            print(f"🏆 Melhor desempenho CPU: {best_cpu['speedup']:.2f}x ({best_cpu['method']} - {best_cpu['threads']} threads)")
            
            if len(cpu_data) > 1:
                efficiency_analysis = cpu_data.groupby('threads')['efficiency'].mean()
                print(f"📈 Eficiência média por threads: {efficiency_analysis.to_dict()}")
        
        if not gpu_data.empty:
            best_gpu = gpu_data.loc[gpu_data['speedup'].idxmax()]
            print(f"🚀 Melhor desempenho GPU: {best_gpu['speedup']:.2f}x ({best_gpu['method']})")
            
            if not cpu_data.empty:
                advantage = best_gpu['speedup'] / cpu_data['speedup'].max()
                print(f"⚡ Vantagem da GPU sobre CPU: {advantage:.2f}x")

def main():
    """Função principal"""
    print("Iniciando benchmark comparativo completo...")
    
    # Verifica dependências
    missing_deps = []
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        print("❌ Dependências faltando:")
        for dep in missing_deps:
            print(f"   {dep}")
        print("\nInstale com: pip install -r requirements.txt")
        return
    
    # Executa benchmark
    benchmark = CompleteMatrixBenchmark(matrix_size=2000)
    results = benchmark.run_complete_benchmark()
    
    # Salva e visualiza resultados
    benchmark.save_complete_results(results)
    benchmark.create_comprehensive_plots(results)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETO FINALIZADO!")
    print("📁 Arquivos gerados:")
    print("   - complete_benchmark_results.csv")
    print("   - complete_benchmark_results.png")
    print("="*80)

if __name__ == "__main__":
    main() 