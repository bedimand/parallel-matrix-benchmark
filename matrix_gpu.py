#!/usr/bin/env python3
"""
Processamento Paralelo de Matrizes - GPU (CUDA)
Trabalho de Aceleração em Ciência de Dados usando Computação Paralela

Este programa demonstra o processamento paralelo de matrizes usando GPU CUDA
e compara com o desempenho da CPU.

Autor: Trabalho de Faculdade
Data: 2024
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import sys

# Tentativa de importar bibliotecas CUDA
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy detectado - GPU CUDA disponível")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy não encontrado - instalando...")

try:
    from numba import cuda, float32
    import numba
    NUMBA_CUDA_AVAILABLE = True
    print("Numba CUDA detectado")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    print("Numba CUDA não encontrado")

class MatrixProcessorGPU:
    def __init__(self, matrix_size=2000):
        """
        Inicializa o processador de matrizes GPU
        
        Args:
            matrix_size (int): Tamanho das matrizes quadradas a serem processadas
        """
        self.matrix_size = matrix_size
        self.check_gpu_availability()
        print(f"Testando matrizes de tamanho: {matrix_size}x{matrix_size}")
        
    def check_gpu_availability(self):
        """Verifica disponibilidade da GPU"""
        self.gpu_available = False
        
        if CUPY_AVAILABLE:
            try:
                # Testa se a GPU está realmente disponível
                cp.cuda.runtime.getDeviceCount()
                device = cp.cuda.Device(0)
                print(f"GPU detectada: RTX 3070 Ti (Device {device.id})")
                print(f"Memória GPU: {cp.cuda.MemInfo()[1] // (1024**3)} GB")
                self.gpu_available = True
                self.gpu_type = 'cupy'
            except Exception as e:
                print(f"Erro ao acessar GPU com CuPy: {e}")
        
        if NUMBA_CUDA_AVAILABLE and not self.gpu_available:
            try:
                if cuda.is_available():
                    gpu = cuda.get_current_device()
                    print(f"GPU detectada via Numba: {gpu.name}")
                    print(f"Compute Capability: {gpu.compute_capability}")
                    self.gpu_available = True
                    self.gpu_type = 'numba'
            except Exception as e:
                print(f"Erro ao acessar GPU com Numba: {e}")
        
        if not self.gpu_available:
            print("⚠️  AVISO: GPU CUDA não disponível. Executando apenas comparação CPU.")
            self.gpu_type = None
    
    def generate_matrices(self):
        """Gera duas matrizes aleatórias para teste"""
        np.random.seed(42)  # Para resultados reproduzíveis
        matrix_a = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        matrix_b = np.random.random((self.matrix_size, self.matrix_size)).astype(np.float32)
        return matrix_a, matrix_b
    
    def cpu_multiplication(self, matrix_a, matrix_b):
        """
        Multiplicação de matrizes na CPU (baseline)
        """
        print("Executando multiplicação na CPU...")
        start_time = time.time()
        
        result = np.dot(matrix_a, matrix_b)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Tempo CPU: {execution_time:.4f} segundos")
        return result, execution_time
    
    def cupy_multiplication(self, matrix_a, matrix_b):
        """
        Multiplicação de matrizes usando CuPy (GPU)
        """
        if not CUPY_AVAILABLE:
            return None, float('inf')
            
        print("Executando multiplicação na GPU (CuPy)...")
        
        # Transfere dados para GPU
        start_transfer = time.time()
        gpu_a = cp.asarray(matrix_a)
        gpu_b = cp.asarray(matrix_b)
        transfer_time = time.time() - start_transfer
        
        # Executa multiplicação na GPU
        start_compute = time.time()
        gpu_result = cp.dot(gpu_a, gpu_b)
        cp.cuda.Stream.null.synchronize()  # Aguarda conclusão
        compute_time = time.time() - start_compute
        
        # Transfere resultado de volta
        start_back = time.time()
        result = cp.asnumpy(gpu_result)
        back_time = time.time() - start_back
        
        total_time = transfer_time + compute_time + back_time
        
        print(f"Tempo GPU (CuPy): {total_time:.4f} segundos")
        print(f"  - Transferência para GPU: {transfer_time:.4f}s")
        print(f"  - Computação GPU: {compute_time:.4f}s")
        print(f"  - Transferência de volta: {back_time:.4f}s")
        
        return result, total_time, compute_time
    
    def numba_matrix_multiply(self, A, B, C):
        """
        Kernel CUDA para multiplicação de matrizes usando Numba
        """
        if not NUMBA_CUDA_AVAILABLE:
            return None
            
        @cuda.jit
        def _kernel(A, B, C):
            row, col = cuda.grid(2)
            
            if row < C.shape[0] and col < C.shape[1]:
                tmp = 0.0
                for k in range(A.shape[1]):
                    tmp += A[row, k] * B[k, col]
                C[row, col] = tmp
        
        return _kernel
    
    def numba_multiplication(self, matrix_a, matrix_b):
        """
        Multiplicação de matrizes usando Numba CUDA
        """
        if not NUMBA_CUDA_AVAILABLE:
            return None, float('inf')
            
        print("Executando multiplicação na GPU (Numba CUDA)...")
        
        # Prepara arrays
        start_time = time.time()
        
        # Aloca memória na GPU
        d_a = cuda.to_device(matrix_a)
        d_b = cuda.to_device(matrix_b)
        d_c = cuda.device_array((self.matrix_size, self.matrix_size), dtype=np.float32)
        
        # Configura grid e block
        threads_per_block = (16, 16)
        blocks_per_grid_x = (self.matrix_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (self.matrix_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Executa kernel
        kernel = self.numba_matrix_multiply(None, None, None)
        if kernel is None:
            return None, float('inf')
        kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
        
        # Copia resultado de volta
        result = d_c.copy_to_host()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Tempo GPU (Numba): {execution_time:.4f} segundos")
        
        return result, execution_time
    
    def benchmark_all_methods(self):
        """
        Executa benchmark comparando CPU vs GPU
        """
        print("="*60)
        print("BENCHMARK - PROCESSAMENTO PARALELO DE MATRIZES (CPU vs GPU)")
        print("="*60)
        
        # Gera matrizes
        matrix_a, matrix_b = self.generate_matrices()
        
        results = {
            'method': [],
            'time': [],
            'speedup': [],
            'compute_only_time': []
        }
        
        # Teste CPU (baseline)
        cpu_result, cpu_time = self.cpu_multiplication(matrix_a, matrix_b)
        baseline_time = cpu_time
        
        results['method'].append('CPU (NumPy)')
        results['time'].append(cpu_time)
        results['speedup'].append(1.0)
        results['compute_only_time'].append(cpu_time)
        
        print("\n" + "="*60)
        print("TESTES GPU")
        print("="*60)
        
        # Teste CuPy
        if self.gpu_available and self.gpu_type == 'cupy':
            gpu_result, gpu_time, gpu_compute_time = self.cupy_multiplication(matrix_a, matrix_b)
            
            if gpu_result is not None:
                speedup = baseline_time / gpu_time
                compute_speedup = baseline_time / gpu_compute_time
                
                results['method'].append('GPU (CuPy - Total)')
                results['time'].append(gpu_time)
                results['speedup'].append(speedup)
                results['compute_only_time'].append(gpu_compute_time)
                
                results['method'].append('GPU (CuPy - Só Computação)')
                results['time'].append(gpu_compute_time)
                results['speedup'].append(compute_speedup)
                results['compute_only_time'].append(gpu_compute_time)
                
                print(f"Speedup total: {speedup:.2f}x")
                print(f"Speedup apenas computação: {compute_speedup:.2f}x")
                
                # Verifica se os resultados são similares
                if np.allclose(cpu_result, gpu_result, rtol=1e-5):
                    print("✅ Resultados CPU e GPU são consistentes")
                else:
                    print("⚠️  Diferença detectada entre resultados CPU e GPU")
        
        # Teste Numba
        if self.gpu_available and NUMBA_CUDA_AVAILABLE:
            try:
                numba_result, numba_time = self.numba_multiplication(matrix_a, matrix_b)
                
                if numba_result is not None:
                    speedup = baseline_time / numba_time
                    
                    results['method'].append('GPU (Numba CUDA)')
                    results['time'].append(numba_time)
                    results['speedup'].append(speedup)
                    results['compute_only_time'].append(numba_time)
                    
                    print(f"Speedup Numba: {speedup:.2f}x")
                    
                    # Verifica se os resultados são similares
                    if np.allclose(cpu_result, numba_result, rtol=1e-5):
                        print("✅ Resultados CPU e Numba GPU são consistentes")
                    else:
                        print("⚠️  Diferença detectada entre resultados CPU e Numba GPU")
            except Exception as e:
                print(f"Erro no teste Numba: {e}")
        
        return results
    
    def plot_results(self, results):
        """
        Cria gráficos dos resultados
        """
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = results['method']
        times = results['time']
        speedups = results['speedup']
        
        # Gráfico 1: Tempo de execução
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        bars1 = ax1.bar(methods, times, color=colors[:len(methods)], alpha=0.7)
        ax1.set_ylabel('Tempo de Execução (s)')
        ax1.set_title('Tempo de Execução por Método')
        ax1.tick_params(axis='x', rotation=45)
        
        # Adiciona valores nas barras
        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Gráfico 2: Speedup
        bars2 = ax2.bar(methods, speedups, color=colors[:len(methods)], alpha=0.7)
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup Relativo à CPU')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline CPU')
        
        # Adiciona valores nas barras
        for bar, speedup_val in zip(bars2, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{speedup_val:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gpu_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(self, results):
        """
        Salva os resultados em um arquivo CSV
        """
        df = pd.DataFrame(results)
        df.to_csv('gpu_benchmark_results.csv', index=False)
        print(f"\nResultados salvos em: gpu_benchmark_results.csv")
        
        # Mostra tabela resumo
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))

def install_cupy():
    """
    Tenta instalar CuPy automaticamente
    """
    try:
        import subprocess
        print("Tentando instalar CuPy...")
        
        # Detecta versão CUDA
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Instala CuPy para CUDA
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'], 
                          check=True)
            print("CuPy instalado com sucesso!")
            return True
        else:
            print("CUDA não detectado. Instalando CuPy CPU...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'cupy-cpu'], 
                          check=True)
            return False
            
    except Exception as e:
        print(f"Erro ao instalar CuPy: {e}")
        return False

def main():
    """
    Função principal
    """
    print("Iniciando benchmark de processamento paralelo de matrizes (GPU)")
    print(f"Sistema: {psutil.virtual_memory().total // (1024**3)} GB RAM")
    
    # Tenta instalar CuPy se não estiver disponível
    if not CUPY_AVAILABLE:
        cupy_installed = install_cupy()
        if cupy_installed:
            print("Reinicie o programa para usar CuPy")
            return
    
    # Inicializa o processador
    processor = MatrixProcessorGPU(matrix_size=2000)
    
    if not processor.gpu_available:
        print("\n⚠️  GPU não disponível. Execute este comando para verificar:")
        print("nvidia-smi")
        print("\nPara instalar CUDA e CuPy:")
        print("pip install cupy-cuda12x  # Para CUDA 12.x")
        print("pip install cupy-cuda11x  # Para CUDA 11.x")
    
    # Executa benchmark
    results = processor.benchmark_all_methods()
    
    # Salva resultados
    processor.save_results_to_csv(results)
    
    # Cria gráficos
    processor.plot_results(results)
    
    print("\n" + "="*60)
    print("BENCHMARK CONCLUÍDO!")
    print("Arquivos gerados:")
    print("- gpu_benchmark_results.csv")
    print("- gpu_benchmark_results.png")
    print("="*60)

if __name__ == "__main__":
    main() 