import numpy as np
import faiss

def cx_simulated_binary_w_cx_uniform(ind1, ind2, eta, indpb, is_nominal):
    """
    유전 알고리즘에서 교차 연산을 수행하는 함수
    범주형 변수와 실수 변수를 모두 처리할 수 있도록 수정

    eta : cx_simulated_binary 파라미터
    indpb : 변수별 변이 확률
    is_nominal : 변수가 범주형인지 여부
    """

    is_nominal = np.array(is_nominal, dtype=bool)

    rand_uniform = np.random.random(len(ind1))
    rand_pb = np.random.random(len(ind1))

    # nominal 값
    mask_nominal = (is_nominal & (rand_pb < indpb))
    ind1[mask_nominal], ind2[mask_nominal] = ind2[mask_nominal], ind1[mask_nominal]

    # 실수 값
    mask_real = ~is_nominal
    rand_real = rand_uniform[mask_real]
    
    beta = np.where(
        rand_real < 0.5,
        (2. * rand_real) ** (1. / (eta + 1.)),
        (1. / (2. * (1. - rand_real))) ** (1. / (eta + 1.))
    )
    
    
    ind1_real = ind1[mask_real]
    ind2_real = ind2[mask_real]
    ind1[mask_real] = 0.5 * ((1 + beta) * ind1_real + (1 - beta) * ind2_real)
    ind2[mask_real] = 0.5 * ((1 - beta) * ind1_real + (1 + beta) * ind2_real)

    return ind1, ind2


def mutGaussian_mutUniformInt(ind, mu, sigma, indpb, is_nominal):
    """
    유전 알고리즘에서 돌연변이 연산을 수행하는 함수
    범주형 변수와 실수 변수를 모두 처리할 수 있도록 수정

    is_nominal : 변수가 범주형인지 여부
    mu : 변수 평균 if 변수가 연속형 else 변수 최솟값
    sigma : 변수 표준편차 if 변수가 연속형 else 변수 최댓값
    """

    is_nominal = np.array(is_nominal, dtype=bool)

    mask = np.random.rand(len(ind)) < indpb

    # 범주 변수 변이 
    cat_indices = np.where(is_nominal & mask)[0] 
    ind[cat_indices] = np.random.randint(mu[cat_indices], sigma[cat_indices] + 1)

    # 연속형 변수 변이
    cont_indices = np.where(~is_nominal & mask)[0]
    ind[cont_indices] += np.random.normal(mu[cont_indices], sigma[cont_indices])

    return ind,


def lexicographic_selection(population,k):
    """
    개체의 fitness를 내림차순 정렬한 후 상위 k개를 선택합니다.
    
    Args:
        population (list): 평가된 개체 리스트.
        k (int): 선택할 개체 수.
    
    Returns:
        population (list): 선택된 개체 리스트 
    """
    
    population.sort(key=lambda ind: tuple(val * w for val, w in zip(ind.fitness.values, ind.fitness.weights)), 
                    reverse=True)

    return population[:k] 


def kmeans_clustering(population, k):
    """
    GPU 사용 여부에 따라 KMeans 클러스터링을 수행하는 함수.
    GPU 옵션이 지원되지 않으면, CPU 인덱스를 GPU로 변환하는 방식으로 fallback 함.
    Args:
        population (numpy.ndarray): 클러스터링할 데이터 배열
        k (int): 클러스터 개수
    Returns:
        cluster_labels (numpy.ndarray): 각 데이터의 클러스터 index
        centroids (numpy.ndarray): 각 클러스터의 중심점
    """
    n, d = population.shape
    population = population.astype('float32')
    # GPU가 사용 가능한지 확인
    num_gpus = faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0
    use_gpu = num_gpus > 0
    res = None
    try:
        if use_gpu:
            # gpu 옵션이 지원되면 바로 gpu=True로 Kmeans 생성
            kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, gpu=True)
        else:
            # GPU 사용 불가하면 CPU 버전으로 생성
            kmeans = faiss.Kmeans(d, k, niter=20, verbose=False)
    except TypeError as e:
        # gpu 옵션을 지원하지 않는 경우, 예외가 발생하면 fallback
        print("GPU 옵션이 Kmeans 생성 시 지원되지 않음, fallback 수행")
        kmeans = faiss.Kmeans(d, k, niter=20, verbose=False)
    # fallback: CPU 인덱스를 GPU로 변환
    if use_gpu and not hasattr(kmeans, 'gpu') and not getattr(kmeans, 'gpu', False):
        # StandardGpuResources 생성 (메모리 최적화도 가능)
        res = faiss.StandardGpuResources()
        res.setTempMemory(int(0.9 * 32 * 1024 * 1024 * 1024)) 
        res.setDefaultNullStreamAllDevices()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, kmeans.index)
        kmeans.index = gpu_index
    kmeans.train(population)
    cluster_labels = kmeans.index.search(population, 1)[1].flatten()
    return cluster_labels, kmeans.centroids

def k_means_selection(population, k):
    """
    Args:
        population (list)
        k (int): K-means에서 나눌 클러스터 개수.

    Returns:
        list: 선택된 ind 리스트.
    """

    adjustment_flag = False  # 클러스터 크기가 홀수일 때 조절하는 플래그
    population_array = np.array(population)
    
    cluster_labels, _ = kmeans_clustering(population_array, k=k)
    
    selected = []
    
    for cluster_id in range(k):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_population = [population[idx] for idx in cluster_indices]
        cluster_size = len(cluster_population)
        
        # 홀수일 떄 개체수 줄어듦 방지 
        selection_size = cluster_size // 2
        if cluster_size % 2 == 1 and not adjustment_flag:
            selection_size += 1
            adjustment_flag = True
        elif cluster_size % 2 == 1 and adjustment_flag:
            adjustment_flag = False
        
        selected.extend(
            lexicographic_selection(
                cluster_population, 
                k=selection_size
              )
        )
    # print(len(selected))
    return selected
