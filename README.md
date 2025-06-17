# Processamento Paralelo de Matrizes - CPU vs GPU

## 📋 Sobre o Projeto

Este projeto implementa um benchmark completo para comparar o desempenho de processamento paralelo de matrizes entre diferentes configurações de CPU e GPU CUDA. Foi desenvolvido como parte do trabalho de **Aceleração em Ciência de Dados usando Computação Paralela**.

## 🎯 Objetivos

- **Aplicar conceitos de paralelismo** em arquiteturas CPU e GPU
- **Comparar paradigmas** de paralelismo de memória compartilhada (OpenMP-like) vs aceleração por GPU (CUDA)
- **Avaliar desempenho e escalabilidade** de diferentes configurações
- **Demonstrar speedup e eficiência** com métricas quantificáveis

## 🏗️ Arquitetura do Projeto

```
Paralela/
├── matrix_cpu.py              # Benchmark CPU (1, 2, 4, 8, todas threads)
├── matrix_gpu.py              # Benchmark GPU (CuPy, Numba CUDA)
├── matrix_comparison.py       # Comparação completa CPU vs GPU
├── requirements.txt           # Dependências do projeto
└── README.md                 # Este arquivo
```

## 📊 Configurações Testadas

### CPU (Paralelismo de Memória Compartilhada)
- **1 thread** (sequencial - baseline)
- **2 threads** 
- **4 threads**
- **8 threads**
- **Todas as threads** disponíveis (20 no seu sistema)

### Implementações CPU
1. **NumPy Paralelo**: Usa bibliotecas otimizadas (BLAS/LAPACK) com controle de threads
2. **Paralelização Manual**: Divide matriz em chunks usando `multiprocessing`

### GPU (Computação Acelerada)
- **CuPy**: Biblioteca NumPy-like para GPU CUDA
- **Numba CUDA**: Kernels CUDA customizados em Python

## 🛠️ Instalação e Configuração

### 1. Dependências Básicas
```bash
pip install -r requirements.txt
```

### 2. Dependências GPU (Opcional)
Para usar aceleração GPU, instale CuPy conforme sua versão CUDA:

```bash
# Verificar versão CUDA
nvcc --version
nvidia-smi

# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x  
pip install cupy-cuda11x

# Sem GPU (apenas para testes)
pip install cupy-cpu
```

### 3. Verificar Sistema
```bash
# CPU
python -c "import multiprocessing; print(f'CPU Cores: {multiprocessing.cpu_count()}')"

# GPU
nvidia-smi
```

## 🚀 Como Executar

### Benchmark Individual CPU
```bash
python matrix_cpu.py
```
**Saída:**
- `cpu_benchmark_results.csv`: Dados tabulados
- `cpu_benchmark_results.png`: Gráficos de desempenho

### Benchmark Individual GPU
```bash
python matrix_gpu.py
```
**Saída:**
- `gpu_benchmark_results.csv`: Dados de comparação CPU vs GPU
- `gpu_benchmark_results.png`: Visualizações

### Benchmark Completo (Recomendado)
```bash
python matrix_comparison.py
```
**Saída:**
- `complete_benchmark_results.csv`: Todos os resultados
- `complete_benchmark_results.png`: Análise visual completa

## 📈 Métricas Avaliadas

### Speedup
```
Speedup = Tempo_Sequencial / Tempo_Paralelo
```

### Eficiência
```
Eficiência = Speedup / Número_de_Threads
```

### Análises Incluídas
- **Tempo de execução** absoluto
- **Speedup** relativo ao baseline sequencial
- **Eficiência** de paralelização
- **Escalabilidade** (análise de crescimento)
- **Comparação CPU vs GPU** (total e apenas computação)

## 📊 Resultados Esperados

### CPU
- **Speedup ideal**: Linear até limite físico de cores
- **Eficiência decrescente**: Devido a overhead e contenção de memória
- **Melhor configuração**: Geralmente entre 4-8 threads para matrizes 2000x2000

### GPU
- **Alto speedup**: 10x-100x+ para computação pura
- **Overhead de transferência**: Reduz speedup total
- **Vantagem**: Maior para operações intensivas em ponto flutuante

## 🔧 Configurações Personalizáveis

### Tamanho da Matriz
```python
# No código, modifique:
processor = MatrixProcessorCPU(matrix_size=1000)  # Para 1000x1000
processor = MatrixProcessorGPU(matrix_size=4000)  # Para 4000x4000
```

### Threads a Testar
```python
# Em matrix_cpu.py, linha ~116:
thread_configs = [1, 2, 4, 8, 16, 32]  # Personalize aqui
```

## 🎓 Alinhamento com Trabalho Acadêmico

Este projeto atende perfeitamente aos requisitos do trabalho:

### ✅ Conceitos Explorados
- **MPI-like**: Paralelização manual com `multiprocessing`
- **OpenMP-like**: Controle de threads NumPy/BLAS
- **CUDA**: Aceleração GPU com CuPy e Numba

### ✅ Objetivos Atendidos
- **Aplicação de paralelismo**: ✅ Múltiplas implementações
- **Diferenciação de paradigmas**: ✅ CPU vs GPU claramente separados
- **Implementação GPU**: ✅ CuPy e Numba CUDA
- **Avaliação de desempenho**: ✅ Métricas quantificáveis
- **Uso de IA**: ✅ Documentado neste README

### ✅ Deliverables
- **Código fonte**: ✅ Três arquivos principais bem documentados
- **Análise de desempenho**: ✅ CSV e gráficos automáticos
- **Comparação**: ✅ Sequencial vs paralelo vs GPU
- **Documentação**: ✅ README completo

## 🤖 Uso de IA no Projeto

Este projeto foi desenvolvido com assistência significativa de IA (Claude/ChatGPT) nas seguintes áreas:

### Assistência de Código
- **Estruturação das classes** e organização modular
- **Implementação de benchmarks** e medição de tempo
- **Paralelização manual** com multiprocessing
- **Integração CUDA** com CuPy e Numba

### Otimização e Boas Práticas
- **Controle de threads** via variáveis de ambiente
- **Tratamento de erros** e verificação de disponibilidade GPU
- **Visualizações** com matplotlib e análise estatística
- **Documentação** e comentários explicativos

### Debugging e Validação
- **Verificação de consistência** entre resultados CPU e GPU
- **Tratamento de dependências** opcionais
- **Instalação automática** de bibliotecas quando possível

## 📋 Checklist de Execução

Para garantir execução completa do trabalho:

- [ ] **Sistema verificado**: CPU cores e GPU disponível
- [ ] **Dependências instaladas**: `pip install -r requirements.txt`
- [ ] **GPU configurada** (se disponível): CuPy instalado
- [ ] **Benchmark CPU executado**: `python matrix_cpu.py`
- [ ] **Benchmark GPU executado**: `python matrix_gpu.py` 
- [ ] **Análise completa**: `python matrix_comparison.py`
- [ ] **Resultados salvos**: CSVs e PNGs gerados
- [ ] **Análise interpretada**: Speedup e eficiência compreendidos

## 📚 Conceitos Acadêmicos Demonstrados

1. **Lei de Amdahl**: Limitações teóricas do paralelismo
2. **Escalabilidade**: Como performance varia com recursos
3. **Overhead**: Custos de paralelização e transferência GPU
4. **Eficiência Energética**: GPU vs CPU para operações massivas
5. **Memory Bound vs Compute Bound**: Diferentes gargalos de performance

---

**Desenvolvido para o curso de Aceleração em Ciência de Dados usando Computação Paralela**  
*Com assistência de IA conforme diretrizes do trabalho* 