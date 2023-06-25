def test_initialize_population():
    nsga2 = NSGA2(population_size=100,
                  num_generations=200,
                  num_objectives=2,
                  num_variables=10,
                  variable_range=(0, 1),
                  mutation_rate=0.01,
                  crossover_rate=0.9,
                  distribution_index_crossover=15,
                  distribution_index_mutation=20)
    nsga2.initialize_population()

    # Check if the population size if correct
    print(len(nsga2.population))
    assert len(nsga2.population) == nsga2.population_size

    # Check if each individual has the correct number of variables
    for i, individual in enumerate(nsga2.population):
        print(f"ind:{i}", individual.variables)
        assert len(individual.variables) == nsga2.num_variables

        # Check if each variables is within the correct range
        for variable in individual.variables:
            assert nsga2.variable_range[0] <= variable <= nsga2.variable_range[1]

def test_dominates():
    nsga2 = NSGA2(population_size=100,
                  num_generations=100,
                  num_objectives=2,
                  num_variables=10,
                  variable_range=(0, 1),
                  mutation_rate=0.01,
                  crossover_rate=0.9,
                  distribution_index_crossover=15,
                  distribution_index_mutation=20)

    individual1 = Individual(variables=[0.1, 0.2])
    individual1.objectives = [1, 2]
    individual2 = Individual(variables=[0.3, 0.4])
    individual2.objectives = [2, 1]

    # Check if individual1 dominates individual2
    assert nsga2.dominates(individual1.objectives, individual2.objectives) == False
    # Check if individual2 dominates individual1
    assert nsga2.dominates(individual2.objectives, individual1.objectives) == False

    individual3 = Individual(variables=[0.5, 0.6])
    individual3.objectives = [1, 1]
    individual4 = Individual(variables=[0.7, 0.8])
    individual4.objectives = [2, 2]

    # Check if individual3 dominates individual4
    assert nsga2.dominates(individual3.objectives, individual4.objectives) == True
    # Check if individual4 dominates individual3
    assert nsga2.dominates(individual4.objectives, individual3.objectives) == False


def test_fast_nondominated_sort():
    nsga2 = NSGA2(population_size=100,
                  num_generations=100,
                  num_objectives=2,
                  num_variables=10,
                  variable_range=(0, 1),
                  mutation_rate=0.01,
                  crossover_rate=0.9,
                  distribution_index_crossover=15,
                  distribution_index_mutation=20)

    individual1 = Individual(variables=[0.1, 0.2])
    individual1.objectives = [1, 2]
    individual2 = Individual(variables=[0.3, 0.4])
    individual2.objectives = [2, 1]
    individual3 = Individual(variables=[0.5, 0.6])
    individual3.objectives = [1, 1]
    individual4 = Individual(variables=[0.7, 0.8])
    individual4.objectives = [2, 2]

    nsga2.population = [individual1, individual2, individual3, individual4]
    fronts = nsga2.fast_nondominated_sort()
    #for i, front in enumerate(fronts):
    #    print(f"{i}: {front[:]}")

    # Check if the first front is correct
    assert fronts[0] == [individual3]
    
    # Check if the second front is correct
    assert fronts[1] == [individual2, individual1]
    #assert fronts[1] == [individual1, individual2]

    # Check if the third front is correct
    assert fronts[2] == [individual4]


def test_calculate_crowding_distance():
    nsga2 = NSGA2(population_size=100,
                  num_generations=100,
                  num_objectives=2,
                  num_variables=10,
                  variable_range=(0, 1),
                  mutation_rate=0.01,
                  crossover_rate=0.9,
                  distribution_index_crossover=15,
                  distribution_index_mutation=20)

    individual1 = Individual(variables=[0.1, 0.2])
    individual1.objectives = [1, 2]
    individual2 = Individual(variables=[0.3, 0.4])
    individual2.objectives = [2, 1]
    individual3 = Individual(variables=[0.5, 0.6])
    individual3.objectives = [1, 1]
    individual4 = Individual(variables=[0.7, 0.8])
    individual4.objectives = [2, 2]
    individual5 = Individual(variables=[0.9, 1.0])
    individual5.objectives = [1.5, 1.5]

    # フロントの生成
    front = [individual1, individual2, individual3, individual4, individual5]
    # Crowding distanceの計算
    nsga2.calculate_crowding_distance(front)

    # フロントの個体それぞれのcrowding distanceを確認する
    # この段階では、る論との両端の個体のcrowding distanceだけが無限大になるべき
    for individual in front:
        print(f"Individual {individual.variables} - Crowding Distance: {individual.crowding_distance}")

    # Check if the crowding distance is correctly calculated
    #for individual in front:
    #    print(individual.crowding_distance)
    #    assert individual.crowding_distance >= 0

def test_selection():
    nsga2 = NSGA2(population_size=100,
                  num_generations=100,
                  num_objectives=2,
                  num_variables=10,
                  variable_range=(0, 1),
                  mutation_rate=0.01,
                  crossover_rate=0.9,
                  distribution_index_crossover=15,
                  distribution_index_mutation=20)

    individual1 = Individual(variables=[0.1, 0.2])
    #individual1.objectives = [1, 2]
    individual1.rank = 1
    individual1.crowding_distance = 0.5

    individual2 = Individual(variables=[0.3, 0.4])
    #individual2.objectives = [2, 1]
    individual2.rank = 2
    individual2.crowding_distance = 0.8

    individual3 = Individual(variables=[0.5, 0.6])
    #individual3.objectives = [1, 1]
    individual3.rank = 1
    individual3.crowding_distance = 0.7

    individual4 = Individual(variables=[0.7, 0.8])
    #individual4.objectives = [2, 2]
    individual4.rank = 3
    individual4.crowding_distance = 0.3

    nsga2.population = [individual1, individual2, individual3, individual4]
    selected_population = nsga2.selection()

    # Check if the selected individuals have valid rank and crowding distance
    for individual in selected_population:
        assert individual.rank is not None
        assert individual.crowding_distance is not None
        print(f"rank:{individual.rank}, CD:{individual.crowding_distance}")

