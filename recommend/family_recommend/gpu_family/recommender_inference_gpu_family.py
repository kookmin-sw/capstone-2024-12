from utils_inference_family import *

def get_cheapest_instance_set_per_memory(df):
    # df 에는 미리 전처리가 되어있어야 합니다.
    # e.g.) SPS 2이상의 인스턴스들만 필터링 되어있고 price와 해당 인스턴스의 spec 등이 미리 병합되어 있어야 합니다.
    
    instance_set = set()
    for memory in df['MemoryGiB'].unique():
        df_great_equal = df[df['MemoryGiB'] >= memory]
        for i, row in df_great_equal[df_great_equal['SpotPrice'] <= df_great_equal['SpotPrice'].min()].iterrows():
            instance_set.add(row['InstanceType'])
    
    return instance_set

# 해당 리전에 대해 Karpenter 가 스케줄링 할 Instance 목록을 제공합니다.
def get_gpu_instance_family_for_inference(region):
    price_df = get_price_df(region)
    price_df = price_df[price_df['SPS'] >= 2]
    instance_df = get_instance_df(region)

    df = pd.merge(price_df, instance_df, on=['Region', 'InstanceType'], how='inner')
    # architecture 가 x86 인 것만 제공
    df_exploded = df.explode('SupportedArchitectures')
    df = df.loc[df_exploded[df_exploded['SupportedArchitectures'].isin(['x86_64'])].index.unique()]
    # NVIDIA GPU 모델의 경우만 제공합니다. 즉, Radeon 등 다른 제조회사의 GPU 인스턴스는 제외합니다.
    df = df[df['GPUManufacturer'] == 'NVIDIA'].reset_index(drop=True)
    
    family = set()

    for gpu_memory in df['TotalGPUMemoryGiB'].unique():
        df_great_equal_gpu_memory = df[df['TotalGPUMemoryGiB'] >= gpu_memory]
        family_per_gpu_memory = get_cheapest_instance_set_per_memory(df_great_equal_gpu_memory)
        
        family |= family_per_gpu_memory
    
    return list(family)