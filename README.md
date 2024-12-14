# **GADEMO API**

### **Descrição**

A **GADEMO** é uma plataforma voltada para a execução de algoritmos genéticos, desenvolvida com foco na modularidade e na escalabilidade. Este sistema é projetado para otimizar funções matemáticas e gerar análises detalhadas sobre o desempenho de diferentes configurações de algoritmos genéticos.

A API foi construída em Python, utilizando práticas modernas de programação, como o padrão **Domain-Driven Design (DDD)**.

---

### **Funcionalidades**

- Suporte a múltiplos tipos de operadores genéticos (crossover, mutação e seleção).
- Configurações customizáveis, incluindo tamanhos de população, taxas de crossover e mutação.
- Suporte ao modo **Steady-State**, com ou sem duplicados, incluindo controle de GAP.
- Normalização linear para adaptação de fitness.
- Avaliação de funções matemáticas de benchmark como **Rastrigin**, **Ackley**, **Drop-Wave**, entre outras.
- Documentação da API acessível via Swagger.

---

### **Estrutura do Projeto**

A estrutura do projeto segue o padrão **Domain-Driven Design (DDD)** para facilitar o desenvolvimento e manutenção. Os principais diretórios incluem:

```
src/
├── config/                # Arquivos de configuração (não utilizado no momento)
├── domain/                # Regras de domínio da aplicação
│   ├── crossover_type.py  # Configurações dos tipos de crossover
│   ├── evaluation.py      # Funções de avaliação para fitness
│   ├── execution_characteristics.py # Características da execução
├── infrastructure/        # Implementação de algoritmos genéticos
│   ├── genetic_algorithm_executor.py # Executor do algoritmo genético
├── interface/             # Camada de interface (API)
│   ├── api.py             # Rotas da API
│   ├── startup.py         # Inicialização do servidor
├── services/              # Serviços de backend
│   ├── genetic_algorithm_service.py # Serviço principal para execução
```

---

### **Requisitos**

Certifique-se de que os seguintes requisitos estejam atendidos antes de rodar o projeto:

- **Python 3.8 ou superior**
- **virtualenv** (ou outra ferramenta de gerenciamento de ambientes virtuais, como pipenv)
- Dependências listadas em `requirements.txt`

---

### **Configuração e Instalação**

#### **Passo 1: Clonar o Repositório**

```bash
git clone https://github.com/seu-usuario/gademo-api.git
cd gademo-api
```

#### **Passo 2: Criar e Ativar um Ambiente Virtual**

Com **virtualenv**:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

Ou com **pipenv**:

```bash
pipenv install
pipenv shell
```

#### **Passo 3: Instalar Dependências**

Usando `pip`:

```bash
pip install -r requirements.txt
```

#### **Passo 4: Executar a API**

Navegue até o diretório correto e inicie o servidor:

```bash
cd src/interface
uvicorn api:app --reload
```

A API estará disponível em: `http://127.0.0.1:8000`

---

### **Documentação**

A documentação completa da API está disponível via Swagger em:

```
http://127.0.0.1:8000/docs
```

---

### **Exemplos de Uso**

#### **Endpoint: `/run-genetic-algorithm`**

Executa o algoritmo genético com os parâmetros fornecidos.

- **Método**: POST
- **Exemplo de Requisição**:

```json
{
    "func": "x**2 + y**2",
    "exec_chars": {
        "maximize": false,
        "population_size": 100,
        "num_generations": 50,
        "crossover_rate": 0.65,
        "mutation_rate": 0.1,
        "interval": [-10, 10]
    },
    "cross_type": {
        "one_point": true,
        "two_point": false,
        "uniform": false
    }
}
```

- **Exemplo de Resposta**:

```json
{
    "best_solution": {
        "generation": 50,
        "fitness": 0.001,
        "values": [0.0, 0.0]
    },
    "last_generation_values": [-0.001, 0.001, 0.005]
}
```

---

### **Hospedagem**

A aplicação está hospedada na plataforma Render:

- **Frontend**: [https://link-do-front.render.com](https://link-do-front.render.com)
- **Documentação da API (Swagger)**: [https://link-da-api.render.com](https://link-da-api.render.com)

---

### **Contribuindo**

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do projeto.
2. Crie uma branch para suas alterações: `git checkout -b minha-feature`.
3. Submeta um PR com a descrição detalhada.

---

### **Licença**

Este projeto está licenciado sob a [MIT License](LICENSE).
