#!/usr/bin/env bash
# Atualiza uma instância da API do GAdemo para uma versão publicada (release).
#
# Uso:
#   VENV_DIR=/caminho/do/venv scripts/atualizar.sh v1.2.0     atualiza para a v1.2.0
#   VENV_DIR=/caminho/do/venv scripts/atualizar.sh --voltar   desfaz a última atualização
#
# Variáveis de ambiente:
#   VENV_DIR      (obrigatória) ambiente virtual usado pelo serviço da API.
#   PYTHON        interpretador usado para criar o ambiente. Padrão: o mesmo
#                 Python do ambiente atual (ou python3, se não houver ambiente).
#   RESTART_CMD   comando que reinicia o serviço, por exemplo
#                 "sudo systemctl restart gademo-api". Sem ele, o script só avisa.
#   REMOTE        remoto do git de onde vêm as versões. Padrão: origin.
#   SEM_CONFIRMACAO=1  não pergunta antes de aplicar (para uso automatizado).
#
# O que acontece, nesta ordem:
#   1. baixa as versões publicadas e mostra as notas da versão (CHANGELOG.md);
#   2. ensaio: monta um ambiente separado com o requirements.txt da versão e
#      roda scripts/verificar_instalacao.py nele. Se algo falhar aqui, nada no
#      servidor foi alterado;
#   3. guarda o ambiente atual em VENV_DIR.anterior e anota a versão atual;
#   4. muda o código para a versão, recria VENV_DIR do zero e verifica de novo.
#      Se falhar, volta sozinho ao ambiente e ao código anteriores;
#   5. reinicia o serviço (RESTART_CMD) ou avisa que é hora de reiniciar.
#
# O ambiente é sempre recriado do zero porque "pip install -r" sobre um ambiente
# existente não remove pacotes que saíram do requirements.txt.
#
# Todo o trabalho acontece dentro de funções, e a chamada final fica numa única
# linha. Assim o bash lê o script inteiro antes de começar, o que é necessário
# porque o próprio arquivo muda de versão durante a atualização.

set -euo pipefail

PYTHON_MINIMO="3.12"
ARQ_ANTERIOR=".atualizacao-anterior"
ENSAIO=""

info()  { printf '\n==> %s\n' "$*"; }
aviso() { printf '    %s\n' "$*"; }
erro()  { printf '\nERRO: %s\n' "$*" >&2; }

uso() {
  sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

limpar_ensaio() {
  if [ -n "$ENSAIO" ] && [ -d "$ENSAIO" ]; then
    rm -rf "$ENSAIO"
  fi
}

exigir_venv_dir() {
  if [ -z "${VENV_DIR:-}" ]; then
    erro "Defina VENV_DIR com o caminho do ambiente virtual usado pelo serviço da API."
    return 1
  fi
  case "$VENV_DIR" in
    /*) ;;
    *) VENV_DIR="$CHAMADO_DE/$VENV_DIR" ;;
  esac
  VENV_DIR="${VENV_DIR%/}"
  if [ ! -d "$(dirname "$VENV_DIR")" ]; then
    erro "A pasta onde ficaria o ambiente não existe: $(dirname "$VENV_DIR")."
    return 1
  fi
  # Normaliza o caminho (tira "..") sem resolver um eventual link do próprio ambiente.
  VENV_DIR="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
  if [ "$VENV_DIR" = "$RAIZ" ] || [ "$VENV_DIR" = "/" ]; then
    erro "VENV_DIR inválido: '$VENV_DIR'."
    return 1
  fi
  if [ -e "$VENV_DIR" ] && [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    erro "$VENV_DIR existe mas não parece um ambiente virtual (falta pyvenv.cfg). Por segurança, nada foi feito."
    return 1
  fi
}

escolher_python() {
  if [ -n "${PYTHON:-}" ]; then
    return 0
  fi
  PYTHON="python3"
  if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    local exe
    exe="$(sed -n 's/^executable *= *//p' "$VENV_DIR/pyvenv.cfg" | head -n 1)"
    if [ -n "$exe" ] && [ -x "$exe" ]; then
      PYTHON="$exe"
    fi
  fi
}

checar_python() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    erro "Python não encontrado: $PYTHON. Indique outro com a variável PYTHON."
    return 1
  fi
  local versao
  versao="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
  if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int, '$PYTHON_MINIMO'.split('.'))) else 1)"; then
    erro "A API exige Python $PYTHON_MINIMO ou superior, e $PYTHON é $versao. Indique outro com a variável PYTHON."
    return 1
  fi
  aviso "Python usado:           $PYTHON ($versao)"
}

exigir_arvore_limpa() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    erro "Há alterações locais em arquivos do repositório (veja 'git status'). Resolva antes de continuar."
    return 1
  fi
}

confirmar() {
  if [ "${SEM_CONFIRMACAO:-0}" = "1" ] || [ ! -t 0 ]; then
    return 0
  fi
  local resposta
  read -r -p "Continuar? [s/N] " resposta
  case "$resposta" in
    s|S|sim|SIM) ;;
    *) aviso "Cancelado. Nada foi alterado."; exit 0 ;;
  esac
}

mostrar_notas() {
  local tag="$1" notas
  notas="$(git show "$tag:CHANGELOG.md" 2>/dev/null | awk -v v="${tag#v}" '
    index($0, "## [" v "]") == 1 { p = 1; print; next }
    p && /^## \[/ { exit }
    p && /^\[[^]]*\]: / { next }
    p { print }')"
  if [ -n "$notas" ]; then
    info "Notas da versão $tag"
    printf '%s\n' "$notas" | sed 's/^/    /'
  else
    aviso "(não há notas para $tag no CHANGELOG.md)"
  fi
}

# Cria um ambiente do zero e instala o requirements. $1 = destino, $2 = requirements.
# Cada passo tem "|| return 1" porque o set -e não vale dentro de funções
# chamadas como condição de um if.
criar_ambiente() {
  local destino="$1" requisitos="$2"
  "$PYTHON" -m venv "$destino" || return 1
  "$destino/bin/python" -m pip install --quiet --disable-pip-version-check -r "$requisitos" || return 1
  "$destino/bin/python" -m pip check --disable-pip-version-check || return 1
}

# Roda a verificação da versão. $1 = ambiente, $2 = raiz do código.
verificar() {
  local ambiente="$1" codigo="$2"
  if [ -f "$codigo/scripts/verificar_instalacao.py" ]; then
    "$ambiente/bin/python" "$codigo/scripts/verificar_instalacao.py" || return 1
  else
    (cd "$codigo/src/api" && "$ambiente/bin/python" -c "import main") || return 1
  fi
}

# Volta ao ambiente guardado e ao código anotado. $1 = commit anterior.
# O ambiente que sai é guardado em VENV_DIR.descartado, para inspeção.
restaurar() {
  local commit="$1"
  if [ -e "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR.descartado"
    mv "$VENV_DIR" "$VENV_DIR.descartado"
  fi
  if [ -e "$VENV_DIR.anterior" ]; then
    mv "$VENV_DIR.anterior" "$VENV_DIR"
  fi
  git checkout --quiet "$commit"
  aviso "Código e ambiente voltaram para $(git describe --tags --always)."
}

reiniciar() {
  if [ -n "${RESTART_CMD:-}" ]; then
    info "Reiniciando o serviço: $RESTART_CMD"
    if ! bash -c "$RESTART_CMD"; then
      erro "O comando de reinício falhou. Código e ambiente já estão na nova versão; reinicie o serviço manualmente."
      return 1
    fi
  else
    info "Agora reinicie o serviço da API para carregar a nova versão."
  fi
}

atualizar() {
  local tag="$1"
  exigir_venv_dir
  escolher_python
  exigir_arvore_limpa

  info "Buscando as versões publicadas em $REMOTE"
  git fetch --quiet --tags "$REMOTE"
  if ! git rev-parse -q --verify "refs/tags/$tag^{commit}" >/dev/null; then
    erro "A versão $tag não existe. Últimas publicadas: $(git tag -l 'v*' --sort=-v:refname | head -n 5 | tr '\n' ' ')"
    return 1
  fi

  local atual commit_atual
  atual="$(git describe --tags --always)"
  commit_atual="$(git rev-parse HEAD)"
  aviso "Versão atual do código: $atual"
  aviso "Versão de destino:      $tag"
  aviso "Ambiente virtual:       $VENV_DIR"
  checar_python
  mostrar_notas "$tag"
  confirmar

  info "Ensaio: montando um ambiente separado com as dependências de $tag"
  ENSAIO="$(mktemp -d "$(dirname "$VENV_DIR")/.gademo-ensaio.XXXXXX")"
  trap limpar_ensaio EXIT
  mkdir "$ENSAIO/codigo"
  git archive "$tag" | tar -x -C "$ENSAIO/codigo"
  if ! criar_ambiente "$ENSAIO/venv" "$ENSAIO/codigo/requirements.txt" \
     || ! verificar "$ENSAIO/venv" "$ENSAIO/codigo"; then
    erro "O ensaio falhou. Nada foi alterado no servidor: código e ambiente continuam como estavam."
    return 1
  fi
  limpar_ensaio

  info "Aplicando $tag"
  printf '%s\n' "$commit_atual" > "$ARQ_ANTERIOR"
  git checkout --quiet "$tag"
  if [ -e "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR.anterior"
    mv "$VENV_DIR" "$VENV_DIR.anterior"
  fi
  if ! criar_ambiente "$VENV_DIR" "$RAIZ/requirements.txt" || ! verificar "$VENV_DIR" "$RAIZ"; then
    erro "A instalação definitiva falhou. Restaurando o estado anterior."
    restaurar "$commit_atual"
    rm -f "$ARQ_ANTERIOR"
    return 1
  fi

  reiniciar
  info "Pronto: código e ambiente em $tag."
  aviso "Para conferir: a expressão x.__class__ deve ser recusada com erro 400,"
  aviso "e o endereço /openapi.json da API mostra o número da versão."
  aviso "Para desfazer: VENV_DIR=$VENV_DIR scripts/atualizar.sh --voltar"
}

voltar() {
  exigir_venv_dir
  if [ ! -f "$ARQ_ANTERIOR" ]; then
    erro "Não há atualização registrada para desfazer."
    return 1
  fi
  if [ ! -e "$VENV_DIR.anterior" ]; then
    erro "Não encontrei $VENV_DIR.anterior, então não dá para desfazer automaticamente."
    return 1
  fi
  exigir_arvore_limpa
  local commit
  commit="$(cat "$ARQ_ANTERIOR")"
  info "Desfazendo a última atualização: de $(git describe --tags --always) para $(git describe --tags --always "$commit")"
  confirmar
  restaurar "$commit"
  rm -f "$ARQ_ANTERIOR"
  reiniciar
}

main() {
  CHAMADO_DE="$(pwd)"
  RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$RAIZ"
  REMOTE="${REMOTE:-origin}"

  case "${1:-}" in
    -h|--help) uso ;;
    "") uso; return 1 ;;
    --voltar) voltar ;;
    -*) erro "Opção desconhecida: $1"; uso; return 1 ;;
    *) atualizar "$1" ;;
  esac
}

main "$@"; exit $?
