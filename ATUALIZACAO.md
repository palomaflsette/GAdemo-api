# Atualização de uma instância da API

Este guia é para quem mantém uma cópia da API do GAdemo em servidor, como a do Maxwell (PUC-Rio).

## Como funciona

- Cada versão da API é publicada como uma release no GitHub, com uma tag no formato `vMAIOR.MENOR.CORREÇÃO` (por exemplo, `v1.2.0`). A lista fica em <https://github.com/palomaflsette/GAdemo-api/releases>.
- A atualização é feita sempre a partir de uma tag, nunca da branch `main`, que pode ter trabalho ainda não publicado.
- Antes de atualizar, vale ler a entrada da versão no [CHANGELOG.md](CHANGELOG.md). Quando o `requirements.txt` muda, ela tem uma seção **Dependências** com o que entrou, saiu ou mudou de versão.
- O ambiente virtual é recriado do zero a cada atualização. Instalar por cima de um ambiente existente deixa para trás pacotes que saíram do projeto. Foi isso que causou o conflito entre NumPy e SciPy em outubro de 2026.

## Requisitos

- Python 3.12 ou superior
- git

## Primeira vez: passando a acompanhar as releases

A cópia do Maxwell já roda o código da v1.0.0. Para passar a acompanhar as releases, basta:

```bash
cd /caminho/do/GAdemo-api
git fetch --tags
git checkout v1.0.0
```

Isso só traz a documentação e os scripts. Não é preciso reinstalar nem reiniciar nada.

Depois do `git checkout` de uma tag, o repositório fica fora de qualquer branch ("detached HEAD"), e `git pull` deixa de funcionar. Isso é esperado: as atualizações passam a ser feitas pela tag.

Opcional: o ambiente atual ainda tem pacotes antigos que não são mais usados, como `pandas` e `flask`. Eles não atrapalham. Se quiserem limpar, basta rodar o script abaixo com `v1.0.0`, o que exige reiniciar o serviço.

## Atualizando com o script

O `scripts/atualizar.sh` faz o procedimento completo:

1. baixa as versões publicadas e mostra as notas da versão escolhida;
2. faz um ensaio num ambiente separado: instala as dependências da versão, roda `pip check` e uma verificação da aplicação. Se o ensaio falhar, nada no servidor é alterado;
3. guarda o ambiente atual em `<venv>.anterior`;
4. muda o código para a versão, recria o ambiente do zero e verifica de novo. Se algo falhar, volta sozinho ao estado anterior;
5. reinicia o serviço, se o comando de reinício for informado.

```bash
cd /caminho/do/GAdemo-api
VENV_DIR=/caminho/do/venv scripts/atualizar.sh v1.1.0
```

| Variável | Obrigatória | Para que serve |
|---|---|---|
| `VENV_DIR` | sim | Caminho do ambiente virtual que o serviço da API usa. |
| `PYTHON` | não | Python usado para criar o ambiente. Padrão: o mesmo do ambiente atual. |
| `RESTART_CMD` | não | Comando que reinicia o serviço, por exemplo `sudo systemctl restart gademo-api`. Sem ele, o script só avisa que é hora de reiniciar. |
| `REMOTE` | não | Remoto do git de onde vêm as versões. Padrão: `origin`. |
| `SEM_CONFIRMACAO` | não | Com valor `1`, o script não pergunta antes de aplicar. |

O script se recusa a continuar se houver alterações locais em arquivos do repositório (veja `git status`). Se a instância precisar de alguma configuração própria, o caminho é uma variável de ambiente (como `GADEMO_ALLOWED_ORIGINS`), não uma edição no código. Se faltar alguma variável para a configuração de vocês, é só avisar a mantenedora para incluí-la numa próxima versão.

### Desfazendo uma atualização

```bash
VENV_DIR=/caminho/do/venv scripts/atualizar.sh --voltar
```

Volta o código e o ambiente para o estado anterior à última atualização e reinicia o serviço (ou avisa para reiniciar). O ambiente que saiu fica guardado em `<venv>.descartado`, para inspeção.

## Atualizando manualmente

O procedimento equivalente, sem o script:

```bash
cd /caminho/do/GAdemo-api
git fetch --tags
git checkout v1.1.0                              # a versão desejada
mv /caminho/do/venv /caminho/do/venv.anterior
python3.12 -m venv /caminho/do/venv
/caminho/do/venv/bin/pip install -r requirements.txt
/caminho/do/venv/bin/pip check
/caminho/do/venv/bin/python scripts/verificar_instalacao.py
# reiniciar o serviço da API
```

Para desfazer: apagar o ambiente novo, renomear `venv.anterior` de volta para `venv`, fazer `git checkout` da versão anterior e reiniciar o serviço.

## Conferindo a versão em execução

- O endereço `/openapi.json` da API informa a versão no campo `info.version`. O número muda a cada release a partir da v1.1.0.
- A expressão `x.__class__` enviada pela interface deve ser recusada com erro 400. Erro 500 nesse teste indica que o serviço ainda roda um código anterior à v1.0.0.
- `scripts/verificar_instalacao.py`, rodado com o Python do ambiente do serviço, confere a instalação sem subir o servidor.
