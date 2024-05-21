def sskai_load_data():
  import pickle
  import numpy as np
  import torch
  from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split

  def load_cifar_batch(filename):
    with open(filename, 'rb') as file:
      batch = pickle.load(file, encoding='latin1')
    # 이미지 데이터 재배열: [num_samples, 3, 32, 32]
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

  def create_datasets(data_paths, test_path):
    # 데이터를 로드하고 하나의 큰 훈련 데이터셋으로 결합
    train_features = []
    train_labels = []
    for path in data_paths:
      features, labels = load_cifar_batch(path)
      train_features.append(features)
      train_labels.append(labels)
    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)
    
    # 테스트 데이터 로드
    test_features, test_labels = load_cifar_batch(test_path)

    # numpy 배열을 PyTorch 텐서로 변환
    train_features = torch.tensor(train_features).permute(0, 3, 1, 2).float() / 255.0
    train_labels = torch.tensor(train_labels).long()
    test_features = torch.tensor(test_features).permute(0, 3, 1, 2).float() / 255.0
    test_labels = torch.tensor(test_labels).long()
    
    # TensorDataset 생성
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    return train_dataset, test_dataset
  
  data_dir_path = "./cifar-10/"
  data_paths = [f'{data_dir_path}/data_batch_{i}' for i in range(1, 6)]
  test_path = f'{data_dir_path}/test_batch'

  # 데이터셋 생성
  train_dataset, test_dataset = create_datasets(data_paths, test_path)

  # DataLoader 생성 예
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

  # 데이터셋 합치기
  combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

  # 합친 데이터셋으로 DataLoader 생성
  combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=64, shuffle=True)

  # combined_dataset에서 data와 label을 따로 떼어서 x, y 변수에 할당
  x = []
  y = []
  for data, label in combined_dataset:
    x.append(data)
    y.append(label)

  x = torch.stack(x)
  y = torch.tensor(y)
    
  return x, y