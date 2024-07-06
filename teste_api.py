from flask import Flask, request
from flask_restx import Api, Resource, fields
import random

app = Flask(__name__)
api = Api(app, version='1.0', title='Genetic Algorithm API',
          description='A simple API to run genetic algorithms',
          )

ns = api.namespace('genetic_algorithm',
                   description='Genetic algorithm operations')

# Modelo de parâmetros de entrada
params_model = api.model('Params', {
    'population_size': fields.Integer(required=True, description='Size of the population'),
    'mutation_rate': fields.Float(required=True, description='Mutation rate'),
    'elitism': fields.Boolean(required=True, description='Elitism enabled or not'),
    'generations': fields.Integer(required=True, description='Number of generations')
})


def genetic_algorithm(params):
    population_size = params.get('population_size', 100)
    mutation_rate = params.get('mutation_rate', 0.01)
    elitism = params.get('elitism', False)
    generations = params.get('generations', 100)

    population = [random.random() for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=lambda x: x)

        if elitism:
            elite = population[:int(0.1 * population_size)]
        else:
            elite = []

        selected = population[:int(0.5 * population_size)]

        offspring = []
        while len(offspring) < population_size - len(elite):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = (parent1 + parent2) / 2
            offspring.append(child)

        population = elite + offspring
        population = [indiv + mutation_rate *
                    (random.random() - 0.5) for indiv in population]

    best_solution = min(population)
    return best_solution


@ns.route('/')
class GeneticAlgorithm(Resource):
    @ns.expect(params_model)
    @ns.response(200, 'Success')
    @ns.response(400, 'Validation Error')
    def post(self):
        """Run a genetic algorithm"""
        params = request.json
        result = genetic_algorithm(params)
        return {'best_solution': result}


if __name__ == '__main__':
    app.run(debug=True)
