#!/usr/bin/env python3
"""
Processamento Paralelo de Matrizes - CPU
Trabalho de Aceleração em Ciência de Dados usando Computação Paralela

Este programa demonstra o processamento paralelo de matrizes usando diferentes
números de threads na CPU (1, 2, 4, 8, e todas as threads disponíveis).

Autor: Trabalho de Faculdade
Data: 2024
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import psutil
import os

def _multiply_chunk_worker(args):
    """Função global para multiplicação de chunk (necessária para multiprocessing)"""
    chunk_a, matrix_b, start_row, end_row = args
    return np.dot(chunk_a, matrix_b)

class MatrixProcessorCPU:
    def __init__(self, matrix_size=2000):
        """
        Inicializa o processador de matrizes
        
        Args:
            matrix_size (int): Tamanho das matrizes quadradas a serem processadas
        """
        self.matrix_size = matrix_size
        self.max_threads = mp.cpu_count()
        print(f"CPU detectada com {self.max_threads} threads/cores")
        print(f"Testando matrizes de tamanho: {matrix_size}x{matrix_size}")
        
    def generate_matrices(self):
        """Gera duas matrizes aleatórias para teste"""
        np.random.seed(42)  # Para resultados reproduzíveis
        matrix_a = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        matrix_b = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        return matrix_a, matrix_b
    
    def sequential_multiplication(self, matrix_a, matrix_b):
        """
        Multiplicação sequencial de matrizes (1 thread)
        """
        print("Executando multiplicação sequencial...")
        start_time = time.time()
        
        # Força o NumPy a usar apenas 1 thread
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        result = np.dot(matrix_a, matrix_b)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Tempo sequencial (1 thread): {execution_time:.4f} segundos")
        return result, execution_time
    
    def parallel_multiplication_threads(self, matrix_a, matrix_b, num_threads):
        """
        Multiplicação paralela usando threads específicas do NumPy
        """
        print(f"Executando multiplicação paralela com {num_threads} threads...")
        start_time = time.time()
        
        # Configura o número de threads para bibliotecas de álgebra linear
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        
        result = np.dot(matrix_a, matrix_b)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Tempo paralelo ({num_threads} threads): {execution_time:.4f} segundos")
        return result, execution_time
    
    def manual_parallel_multiplication(self, matrix_a, matrix_b, num_processes):
        """
        Multiplicação paralela manual dividindo o trabalho entre processos
        """
        print(f"Executando multiplicação paralela manual com {num_processes} processos...")
        start_time = time.time()
        
        # Divide a matriz A em chunks
        chunk_size = self.matrix_size // num_processes
        chunks = []
        
        for i in range(num_processes):
            start_row = i * chunk_size
            end_row = (i + 1) * chunk_size if i < num_processes - 1 else self.matrix_size
            chunk_a = matrix_a[start_row:end_row, :]
            chunks.append((chunk_a, matrix_b, start_row, end_row))
        
        # Executa em paralelo
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(_multiply_chunk_worker, chunks))
        
        # Combina os resultados
        result = np.vstack(results)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Tempo paralelo manual ({num_processes} processos): {execution_time:.4f} segundos")
        return result, execution_time
    
    def benchmark_all_configurations(self):
        """
        Executa benchmark com todas as configurações de threads
        """
        print("="*60)
        print("BENCHMARK - PROCESSAMENTO PARALELO DE MATRIZES (CPU)")
        print("="*60)
        
        # Gera matrizes
        matrix_a, matrix_b = self.generate_matrices()
        
        # Configurações de threads para testar
        thread_configs = [1, 2, 4, 8, self.max_threads]
        
        results = {
            'threads': [],
            'numpy_time': [],
            'manual_time': [],
            'speedup_numpy': [],
            'speedup_manual': [],
            'efficiency_numpy': [],
            'efficiency_manual': []
        }
        
        # Teste sequencial (baseline)
        _, sequential_time = self.sequential_multiplication(matrix_a, matrix_b)
        baseline_time = sequential_time
        
        print("\n" + "="*60)
        print("TESTES PARALELOS")
        print("="*60)
        
        for num_threads in thread_configs:
            print(f"\n--- Testando com {num_threads} thread(s) ---")
            
            # Teste com NumPy paralelo
            _, numpy_time = self.parallel_multiplication_threads(matrix_a, matrix_b, num_threads)
            
            # Teste com paralelização manual (apenas se num_threads > 1)
            if num_threads == 1:
                manual_time = sequential_time
            else:
                _, manual_time = self.manual_parallel_multiplication(matrix_a, matrix_b, num_threads)
            
            # Calcula speedup e eficiência
            speedup_numpy = baseline_time / numpy_time
            speedup_manual = baseline_time / manual_time
            efficiency_numpy = speedup_numpy / num_threads
            efficiency_manual = speedup_manual / num_threads
            
            # Armazena resultados
            results['threads'].append(num_threads)
            results['numpy_time'].append(numpy_time)
            results['manual_time'].append(manual_time)
            results['speedup_numpy'].append(speedup_numpy)
            results['speedup_manual'].append(speedup_manual)
            results['efficiency_numpy'].append(efficiency_numpy)
            results['efficiency_manual'].append(efficiency_manual)
            
            print(f"Speedup NumPy: {speedup_numpy:.2f}x")
            print(f"Speedup Manual: {speedup_manual:.2f}x")
            print(f"Eficiência NumPy: {efficiency_numpy:.2f}")
            print(f"Eficiência Manual: {efficiency_manual:.2f}")
        
        return results
    
    def plot_results(self, results):
        """
        Cria gráficos dos resultados
        """
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        threads = results['threads']
        
        # Gráfico 1: Tempo de execução
        ax1.plot(threads, results['numpy_time'], 'bo-', label='NumPy Paralelo', linewidth=2, markersize=8)
        ax1.plot(threads, results['manual_time'], 'ro-', label='Paralelo Manual', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Threads')
        ax1.set_ylabel('Tempo de Execução (s)')
        ax1.set_title('Tempo de Execução vs Número de Threads')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Gráfico 2: Speedup
        ax2.plot(threads, results['speedup_numpy'], 'bo-', label='NumPy Paralelo', linewidth=2, markersize=8)
        ax2.plot(threads, results['speedup_manual'], 'ro-', label='Paralelo Manual', linewidth=2, markersize=8)
        ax2.plot(threads, threads, 'k--', label='Speedup Ideal', alpha=0.7)
        ax2.set_xlabel('Número de Threads')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup vs Número de Threads')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log', base=2)
        
        # Gráfico 3: Eficiência
        ax3.plot(threads, results['efficiency_numpy'], 'bo-', label='NumPy Paralelo', linewidth=2, markersize=8)
        ax3.plot(threads, results['efficiency_manual'], 'ro-', label='Paralelo Manual', linewidth=2, markersize=8)
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Eficiência Ideal')
        ax3.set_xlabel('Número de Threads')
        ax3.set_ylabel('Eficiência')
        ax3.set_title('Eficiência vs Número de Threads')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Gráfico 4: Comparação de desempenho
        x = np.arange(len(threads))
        width = 0.35
        
        ax4.bar(x - width/2, results['speedup_numpy'], width, label='NumPy Paralelo', alpha=0.8)
        ax4.bar(x + width/2, results['speedup_manual'], width, label='Paralelo Manual', alpha=0.8)
        
        ax4.set_xlabel('Configuração de Threads')
        ax4.set_ylabel('Speedup')
        ax4.set_title('Comparação de Speedup por Configuração')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{t} thread{"s" if t > 1 else ""}' for t in threads])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cpu_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(self, results):
        """
        Salva os resultados em um arquivo CSV
        """
        df = pd.DataFrame(results)
        df.to_csv('cpu_benchmark_results.csv', index=False)
        print(f"\nResultados salvos em: cpu_benchmark_results.csv")
        
        # Mostra tabela resumo
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))

def main():
    """
    Função principal
    """
    print("Iniciando benchmark de processamento paralelo de matrizes (CPU)")
    print(f"Sistema: {psutil.virtual_memory().total // (1024**3)} GB RAM")
    print(f"CPU: {mp.cpu_count()} cores/threads")
    
    # Inicializa o processador
    processor = MatrixProcessorCPU(matrix_size=2000)
    
    # Executa benchmark
    results = processor.benchmark_all_configurations()
    
    # Salva resultados
    processor.save_results_to_csv(results)
    
    # Cria gráficos
    processor.plot_results(results)
    
    print("\n" + "="*60)
    print("BENCHMARK CONCLUÍDO!")
    print("Arquivos gerados:")
    print("- cpu_benchmark_results.csv")
    print("- cpu_benchmark_results.png")
    print("="*60)

if __name__ == "__main__":
    main() 