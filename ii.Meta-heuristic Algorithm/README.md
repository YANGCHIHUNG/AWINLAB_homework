# ii. Meta-heuristic Algorithm
## 爬山演算法(Hill climbing, HC)。(20分)
### 1. Initial
隨機產生出一組合法解，並且去評估該組解有多好
```
def ramdomSolution():
    item_status = [ random.randint(0,1) for _ in range(n)]
    total_weight = 0
    is_solution = 0
    profit = 0

    while(is_solution == 0):
        for i in range(n):
            if(item_status[i] == 1):
                total_weight += w[i]
                profit += p[i]

        if(total_weight <= c):
            is_solution = 1
        else: 
            item_status = [ random.randint(0,1) for _ in range(n)]
            total_weight = 0
            profit = 0
            
    return item_status, profit ,total_weight
```
### 2. Transition & 3. Evalution
將第1步產生的解，隨機挑選一個位置進行更新，並且將第2步新產生出的解進行評估。
```
def getNeighbor(solution, profit, weight):
    
    n = len(solution)
    is_solution = False
    
    neightbor_sol = solution.copy()
    neighbor_profit = profit
    neighbor_weight = weight

    while(is_solution is False):
        
        i = random.randint(0,n-1)

        if(neightbor_sol[i] == 0):
            
            neighbor_weight += w[i]

            if(neighbor_weight <= c):
                neightbor_sol[i]=1
                neighbor_profit += p[i]
                is_solution = True
            else:
                neighbor_weight -= w[i]
        else: 
            neightbor_sol[i] = 0
            neighbor_profit -= p[i]
            neighbor_weight -= w[i]
            
            is_solution = True
 
    return neightbor_sol, neighbor_profit, neighbor_weight
```
### 4.Determination

比較第3步評估過後的解是否優於原來的解，若優於原來的解則更新，若無則維持原來的解。
```
def hillClimbing(current_sol, currrent_profit, current_weight):
    
    neightbor_sol, neighbor_profit, neighbor_weight = getNeighbor(current_sol, currrent_profit, current_weight)
    print("\n鄰近解: " + str(neightbor_sol))
    print("鄰近解獲利: " + str(neighbor_profit))
    print("鄰近解總重: " + str(neighbor_weight))

    # 鄰近解若優於或等於先前解則更新
    if(currrent_profit <= neighbor_profit):

        current_sol = neightbor_sol
        currrent_profit = neighbor_profit
        current_weight = neighbor_weight
    
    return current_sol,currrent_profit,current_weight
```
### 將結果繪製成圖表
```
def plotIteration(profit_history, iteration):
    
    iterationTime = iteration

    plt.title('circle of convergence')
    plt.plot(range(1, iterationTime + 1), profit_history, label='Profit')
    plt.xlabel('iteration')
    plt.ylabel('bestProfit')
    plt.legend()
    plt.grid()
    plt.show()
    
# 畫收斂圖
plotIteration(profit_history, iteration)
```
![image](https://hackmd.io/_uploads/BycYmSJ1A.png)

---
## 基因演算法(Genetic algorithm, GA)。(40分) 
### 1. 設定
首先設定交配率(crossover rate)、突變率(mutation)、群體數量(Population size) 以及終止條件。

* 交配率：允許交配的機率。
* 突變率：發生突變的機率。
* 群體數量：一次迭代(Iteration)中會有幾組解。
* 終止條件：Evaluation 次數即為終止條件。
```
#設定初始參數
solutions_per_pop = 50
mutation_rate = 0.4
crossover_rate = 0.8
num_generations = 100
```
### 2. Initial

以隨機方式進行初始化，接著再評估該組解的好壞。

### 3. Transition
#### 3.1 Selection

選擇出一組解，接著再進行兩兩交配。
```
#選擇出一組解，接著再進行兩兩交配。
def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents
```

#### 3.2 Crossover

將選出的那些解兩兩進行交配。
```
#將選出的那些解兩兩進行交配
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings  
```

#### 3.3 Mutation

完成交配環節後，會隨機一個0–1的浮點數，若該浮點數小於設定的突變率則會將隨機一個位置進行轉置，也就是1變0、0變1。若突變率設定過大，可能會導致搜尋策略太過隨機，使演算法無法收斂，進而導致找不到好的解，突變率適合的大小會依照求解問題而有所不同，如何判定突變率有無設定過大、過小，可能就要透過實驗方式才會知道了。只要突變率可達到讓演算法有跳出區域最佳解的目的即可。
```
#隨機選擇一些變數使得0變1、1變0
def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants 
```

### 4.Evaluation

交配突變過後，則將所有新產生的解進行評估。
### 將結果繪製成圖表
![image](https://hackmd.io/_uploads/SJC-Z-gJR.png)
