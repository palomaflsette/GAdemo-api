# Histórico de versões

Todas as mudanças relevantes da API do GAdemo ficam registradas aqui. O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e a numeração segue o [Versionamento Semântico](https://semver.org/lang/pt-BR/) (MAIOR.MENOR.CORREÇÃO).

Sempre que o `requirements.txt` muda, a versão traz uma seção **Dependências** dizendo o que entrou, o que saiu e o que mudou de versão. Instâncias em servidor devem ser atualizadas pelo procedimento de [ATUALIZACAO.md](ATUALIZACAO.md), que recria o ambiente virtual do zero.

## [Não publicado]

Nada ainda.

## [1.0.0] - 2026-10-02

Primeira versão publicada como release. O código da API é o mesmo que já está em produção no Heroku e no Maxwell (commit `8939f22`). Esta versão só acrescenta documentação e as ferramentas de atualização, então quem já está nesse commit não precisa reinstalar nada.

### Segurança

- A expressão enviada para `/run-experiments` é validada antes de qualquer avaliação. Só passam números, `x`, `y`, `pi`, os operadores aritméticos e as funções `sin`, `cos`, `tan`, `sqrt`, `exp` e `log`. Qualquer outra construção é recusada com erro 400. Antes, a expressão ia direto para o `sympify`, que pode executar código Python.
- Expressões com mais de 256 caracteres, constantes acima de 1e6 e torres de expoentes (`a**b**c`) também são recusadas, para evitar travamentos.
- O CORS passa a aceitar apenas as origens do front: `palomasette.com`, `palomaflsette.github.io` e `maxwell.vrac.puc-rio.br` (com e sem `www`, sempre `https`), além de `localhost` para desenvolvimento. Outras origens podem ser liberadas com a variável de ambiente `GADEMO_ALLOWED_ORIGINS`, separadas por vírgula.
- Os parâmetros são validados no servidor: `num_experiments` de 1 a 50, `num_generations` e `population_size` de 1 a 1000, taxas entre 0 e 1 e intervalo dentro de [-1e6, 1e6].
- O contêiner Docker roda sem root e sem `--reload`.

### Dependências

- Todas as versões passam a ser fixadas no `requirements.txt`:
  `deap==1.4.4`, `fastapi==0.142.1`, `gunicorn==26.2.0`, `numpy==2.5.3`, `openpyxl==3.1.5`, `pydantic==2.13.5`, `sympy==1.14.0`, `uvicorn[standard]==0.54.0`.
- **Requer Python 3.12 ou superior** (exigência do NumPy 2.5).
- Pacotes que já fizeram parte do projeto e não são mais usados: `scipy`, `pandas` e `flask` (estavam no `requirements.txt` até julho de 2025). Em ambientes antigos eles podem continuar instalados, porque o `pip install -r` não remove pacotes. O `scipy` antigo, em especial, é incompatível com o NumPy 2.5. Por isso o ambiente deve ser recriado do zero.

### Adicionado

- `CHANGELOG.md`, este arquivo.
- `ATUALIZACAO.md`, com o procedimento de atualização para quem mantém uma instância em servidor.
- `scripts/atualizar.sh`, que atualiza uma instância para uma release recriando o ambiente virtual do zero, com ensaio prévio e opção de desfazer.
- `scripts/verificar_instalacao.py`, que confere uma instalação sem subir o servidor.

### Corrigido

- O README indicava Python 3.8 como versão mínima. A versão mínima é 3.12.

[Não publicado]: https://github.com/palomaflsette/GAdemo-api/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/palomaflsette/GAdemo-api/releases/tag/v1.0.0
