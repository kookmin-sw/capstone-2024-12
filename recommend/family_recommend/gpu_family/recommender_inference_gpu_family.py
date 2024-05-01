from utils_inference_family import *

def get_benchmark(row):
    gpu_model_name = row['GPUModel']
    gpu_count = row['GPUCount']
    benchmark = get_gpu_benchmark(gpu_model_name, gpu_count)
    return benchmark

def get_total_instance_score(row, min_price, max_price, min_benchmark, max_benchmark):
    price = row['SpotPrice']
    benchmark = row['Benchmark']
    sps = row['SPS']

    # 선형 보간을 위한 alpha 값
    # 스팟 가격과 벤치마크 점수를 어떤 비율로 반영할지 결정합니다.
    alpha = 0.7
    price_score = 1 - ((price - min_price) / (max_price - min_price))
    benchmark_score = (benchmark - min_benchmark) / (max_benchmark - min_benchmark)
    # 선형 보간 적용 및 sps 에 높은 비중을 주기 위해서 sps 값을 곱해줍니다.
    total_score = sps * ((1 - alpha) * price_score + alpha * benchmark_score)

    return total_score

# 해당 리전에 대해 Karpenter 가 스케줄링 할 Instance 목록을 제공합니다.
def get_gpu_instance_family_for_inference(region):
    price_df = get_price_df(region)
    #price_df = price_df[price_df['SPS'] >= 2]
    instance_df = get_instance_df(region)

    df = pd.merge(price_df, instance_df, on=['Region', 'InstanceType'], how='inner')

    # benchmark 및 total score 계산
    df['Benchmark'] = df.apply(get_benchmark, axis=1)
    min_price = df['SpotPrice'].min()
    max_price = df['SpotPrice'].max()
    min_benchmark = df['Benchmark'].min()
    max_benchmark = df['Benchmark'].max()
    df['TotalScore'] = df.apply(get_total_instance_score, args=(min_price, max_price, min_benchmark, max_benchmark), axis=1)

    # architecture 가 x86 인 것만 제공
    df_exploded = df.explode('SupportedArchitectures')
    df = df.loc[df_exploded[df_exploded['SupportedArchitectures'].isin(['x86_64'])].index.unique()]
    # NVIDIA GPU 모델의 경우만 제공합니다. 즉, Radeon 등 다른 제조회사의 GPU 인스턴스는 제외합니다.
    df = df[df['GPUManufacturer'] == 'NVIDIA'].reset_index(drop=True)
    df = df[df['GPUCount'] == 1].reset_index(drop=True)
    
    df['MaxTotalScore'] = df.groupby(['InstanceType', 'Region'])['TotalScore'].transform('max')
    # 최종 점수 기준 내림차순 정렬
    sorted_df_by_total_score = df[df['TotalScore'] == df['MaxTotalScore']].sort_values(by='TotalScore', ascending=False)

    family_list = []
    num_of_family = 5
    for index, row in sorted_df_by_total_score.iterrows():
        family_list.append(row['InstanceType'])
        num_of_family -= 1
        if num_of_family <= 0:
            break

    return family_list

if __name__ == "__main__":
    family_list = get_gpu_instance_family_for_inference("ap-northeast-2")