import os
import sys

# Torna os pacotes de src/ (core, domain, application, api) importáveis nos testes.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
