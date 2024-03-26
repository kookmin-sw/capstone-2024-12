from utils import *

def get_recommend_inference(region_name, trained_instance=None, min_cpu_mem=0, min_gpu_mem=0):
    price_df = get_price_df(region_name)
    price_df = price_df[price_df['SPS'] >= 2]
    instance_df = get_instance_df(region_name)
    gpu_df = instance_df[instance_df['GPUCount'] != 0]
    cpu_df = instance_df[instance_df['GPUCount'] == 0].drop(['GPUModel', 'GPUManufacturer', 'GPUCount'], axis=1).reset_index(drop=True)
    
    if trained_instance is not None:
        min_cpu_mem = max(min_cpu_mem, instance_df[instance_df['InstanceType'] == trained_instance]['MemoryGiB'].iloc[0])
        min_gpu_mem = max(min_gpu_mem, instance_df[instance_df['InstanceType'] == trained_instance]['TotalGPUMemoryGiB'].iloc[0])
    
    cpu_recommend = get_cpu_recommend_inference(price_df, cpu_df, min_cpu_mem)
    gpu_recommend = get_gpu_recommend_inference(price_df, gpu_df, min_gpu_mem)
    
    instance_df = instance_df[(instance_df['MemoryGiB'] >= min_cpu_mem) & (instance_df['TotalGPUMemoryGiB'] >= min_gpu_mem)]
    df = pd.merge(price_df, instance_df, on=['InstanceType', 'Region'], how='inner')
    
    min_price_idx = df['SpotPrice'].idxmin()
    recommend_set = set()
    recommend_set.add((df.loc[min_price_idx, 'InstanceType'], df.loc[min_price_idx, 'SpotPrice'], 'cheapest minimum condition'))
    
    return recommend_set.union(cpu_recommend, gpu_recommend)
    
def get_cpu_recommend_inference(price_df, cpu_df, min_cpu_mem):
    df = pd.merge(price_df, cpu_df, on=['InstanceType', 'Region'], how='inner')
    recommend_set = set()
    df['SpotPricePerMem'] = df['SpotPrice'] / df['MemoryGiB']
    spot_price_min_idx_per_memory = df['SpotPricePerMem'].idxmin()
    # cpu memory 당 가장 싼 인스턴스
    recommend_set.add((df.loc[spot_price_min_idx_per_memory, 'InstanceType'], df.loc[spot_price_min_idx_per_memory, 'SpotPrice'], 'cheapest per memory'))
    return recommend_set

def get_gpu_recommend_inference(price_df, gpu_df, min_gpu_mem):
    df = pd.merge(price_df, gpu_df, on=['InstanceType', 'Region'], how='inner')
    df['Benchmark'] = df.apply(lambda row: get_gpu_benchmark(row['GPUModel'], row['GPUCount']), axis=1)
    
    recommend_set = set()
    
    df['SpotPricePerGPUMemory'] = df['SpotPrice'] / df['TotalGPUMemoryGiB']
    df['SpotPricePerBenchmark'] = df['SpotPrice'] / df['Benchmark']
    
    spot_price_min_idx_per_gpu_memory = df['SpotPricePerGPUMemory'].idxmin()
    spot_price_min_idx_per_benchmark = df['SpotPricePerBenchmark'].idxmin()
    
    recommend_per_gpu_memory = (df.loc[spot_price_min_idx_per_gpu_memory, 'InstanceType'], df.loc[spot_price_min_idx_per_gpu_memory, 'SpotPrice'], 'cheapest per gpu memory')
    recommend_per_gpu_benchmark = (df.loc[spot_price_min_idx_per_benchmark, 'InstanceType'], df.loc[spot_price_min_idx_per_benchmark, 'SpotPrice'], 'cheapest per gpu benchmark')
    
    recommend_set.add(recommend_per_gpu_memory)
    recommend_set.add(recommend_per_gpu_benchmark)
    
    return recommend_set