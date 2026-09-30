from pydantic import BaseModel, model_validator, field_validator, Field
from typing import List


class CrossoverType(BaseModel):
    """Define o tipo de crossover, garantindo que apenas um seja selecionado."""
    one_point: bool = True
    two_point: bool = False
    uniform: bool = False

    @model_validator(mode='after')
    def check_single_crossover_type(self) -> 'CrossoverType':
        """Valida que apenas uma das opções de crossover é True."""
        true_options = [
            field for field in self.model_fields
            if getattr(self, field) is True
        ]
        if len(true_options) != 1:
            raise ValueError(
                f"Exatamente um tipo de crossover deve ser selecionado. {len(true_options)} foram selecionados: {true_options}")
        return self


class ExecutionParameters(BaseModel):
    """Agrupa todos os parâmetros e características de uma execução do AG."""
    num_generations: int = Field(..., gt=0, le=1000,
                                  description="Número de gerações (1 a 1000).")
    population_size: int = Field(..., gt=0, le=1000,
                                 description="Tamanho da população (1 a 1000).")
    crossover_rate: float = Field(..., ge=0.0, le=1.0,
                                  description="Taxa de crossover (0.0 a 1.0).")
    mutation_rate: float = Field(..., ge=0.0, le=1.0,
                                 description="Taxa de mutação (0.0 a 1.0).")
    maximize: bool = True
    interval: List[int]
    crossover_type: CrossoverType = Field(default_factory=CrossoverType)

    @field_validator('interval')
    @classmethod
    def check_interval(cls, v: List[int]) -> List[int]:
        """Garante um intervalo [min, max] válido e dentro de limites sãos."""
        if len(v) != 2:
            raise ValueError("interval deve ter exatamente 2 valores: [min, max].")
        low, high = v
        if low >= high:
            raise ValueError("interval[0] (min) deve ser menor que interval[1] (max).")
        if abs(low) > 1_000_000 or abs(high) > 1_000_000:
            raise ValueError("interval fora do limite permitido ([-1e6, 1e6]).")
        return v

    elitism: bool = False
    normalize_linear: bool = False
    normalize_min: int = 0
    normalize_max: int = 100

    steady_state_with_duplicates: bool = False
    steady_state_without_duplicates: bool = False
    gap: float = Field(
        0.0, ge=0.0, le=1.0, description="Gap geracional para Steady-State (0.0 a 1.0).")


    steady_state_removal: str = Field(
        'tournament',
        description="Estratégia de remoção no Steady-State: 'random' (clássico), 'worst' (elitista), 'tournament' (híbrido)"
    )

    @model_validator(mode='after')
    def validate_steady_state_removal(self) -> 'ExecutionParameters':
        """Valida que a estratégia de remoção é válida."""
        valid_strategies = ['random', 'worst', 'tournament']
        if self.steady_state_removal not in valid_strategies:
            raise ValueError(
                f"Estratégia de remoção deve ser uma de: {valid_strategies}. "
                f"Recebido: '{self.steady_state_removal}'"
            )
        return self
