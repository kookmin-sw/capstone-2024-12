from utils_inference_family import *

def get_cheapest_instance_set_per_memory(df):
    # df 에는 미리 전처리가 되어있어야 합니다.
    # e.g.) price와 해당 인스턴스의 spec 등이 미리 병합되어 있어야 합니다.
    
    instance_set = set()
    for memory in df['MemoryGiB'].unique():
        df_great_equal = df[df['MemoryGiB'] >= memory]
        for i, row in df_great_equal[df_great_equal['SpotPrice'] <= df_great_equal['SpotPrice'].min()].iterrows():
            instance_set.add(row['InstanceType'])
    
    return instance_set

# 해당 리전에 대해 Karpenter 가 스케줄링 할 Instance 목록을 제공합니다.
def get_cpu_instance_family_for_inference(region):
    price_df = get_spot_price_df(region)
    instance_df = get_instance_df(region)

    df = pd.merge(price_df, instance_df, on=['InstanceType'], how='inner')
    # architecture 가 x86 인 것만 제공
    df_exploded = df.explode('SupportedArchitectures')
    df = df.loc[df_exploded[df_exploded['SupportedArchitectures'].isin(['x86_64'])].index.unique()]
    # CPU 모델만 제공합니다.
    df = df[df['GPUCount'] == 0].reset_index(drop=True)
    df = df[df['MemoryGiB'] >= 16].reset_index(drop=True)
    
    family = get_cheapest_instance_set_per_memory(df)

    return list(family)

if __name__ == "__main__":
    family_list = get_cpu_instance_family_for_inference("ap-northeast-2")
    print(family_list)