from utils_inference_family import *
import re

def get_benchmark(row):
    gpu_model_name = row['GPUModel']
    gpu_count = row['GPUCount']
    benchmark = get_gpu_benchmark(gpu_model_name, gpu_count)
    return benchmark

def get_spot_price_per_gpu(row):
    gpu_count = row['GPUCount']
    spot_price = row['SpotPrice']
    return spot_price / gpu_count if gpu_count > 0 else spot_price

def get_total_instance_score(row, max_price, max_benchmark,):
    price = row['SpotPricePerGPU']
    benchmark = row['Benchmark']
    
    # gpu_count = row['GPUCount']
    # e_rate = 0.99
    # gpu_count_cost_rate = e_rate ** math.log2(gpu_count)

    # 선형 보간을 위한 alpha 값
    # 스팟 가격 점수와 GPU 점수를 어떤 비율로 반영할지 결정합니다.
    alpha = 0.2
    benchmark_score = benchmark / max_benchmark
    price_score = 1 - price / max_price

    total_score = (1 - alpha) * price_score + alpha * benchmark_score
    # total_score *= gpu_count_cost_rate

    return total_score

def filtering_df_by_family(df, family : list):
    regex = '|'.join(family)
    df = df[df['InstanceType'].str.contains(regex)]
    return df

# 해당 리전에 대해 Karpenter 가 스케줄링 할 Instance 목록을 제공합니다.
def get_family_1_for_inference(region):
    price_df = get_spot_price_df(region)
    instance_df = get_instance_df(region)

    df = pd.merge(price_df, instance_df, on=['InstanceType'], how='left')

    # architecture 가 x86 인 것만 제공
    df_exploded = df.explode('SupportedArchitectures')
    df = df.loc[df_exploded[df_exploded['SupportedArchitectures'].isin(['x86_64'])].index.unique()]
    # NVIDIA GPU 모델의 경우만 제공합니다. 즉, Radeon 등 다른 제조회사의 GPU 인스턴스는 제외합니다.
    df = df[df['GPUManufacturer'] == 'NVIDIA'].reset_index(drop=True)
    #df = df[df['GPUCount'] == 1].reset_index(drop=True)

    # benchmark 및 total score 계산
    df['Benchmark'] = df.apply(get_benchmark, axis=1)
    df['SpotPricePerGPU'] = df.apply(get_spot_price_per_gpu, axis=1)
    min_price = df['SpotPricePerGPU'].min()
    max_price = df['SpotPricePerGPU'].max()
    min_benchmark = df['Benchmark'].min()
    max_benchmark = df['Benchmark'].max()
    df['TotalScore'] = df.apply(get_total_instance_score, args=(max_price, max_benchmark), axis=1)

    df = filtering_df_by_family(df, 
                                ['g3', 'p2',            # group 1
                                 'g4dn', 'g5g', 'p3',   # group 2
                                 'g6', 'gr6', 'g5',     # group 3
                                 'p3dn', 'p4d',         # group 4
                                 'p5'])                 # group 5
    
    df['MaxTotalScore'] = df.groupby(['InstanceType', 'Region'])['TotalScore'].transform('max')
    # 최종 점수 기준 내림차순 정렬
    sorted_df_by_total_score = df[df['TotalScore'] == df['MaxTotalScore']].sort_values(by='TotalScore', ascending=False).reset_index(drop=True)

    family_list = []
    print(sorted_df_by_total_score[['InstanceType', 'vCPU', 'MemoryGiB', 'SpotPrice', 'SpotPricePerGPU', 'GPUModel', 'GPUManufacturer',
               'TotalGPUMemoryGiB', 'Benchmark', 'TotalScore']])

    iterration_count = min(10, len(sorted_df_by_total_score))

    for i in range(iterration_count):
        family_list.append(sorted_df_by_total_score.iloc[i]['InstanceType'])

    return family_list

if __name__ == "__main__":
    family_list = get_family_1_for_inference("us-east-1")
    print(family_list)