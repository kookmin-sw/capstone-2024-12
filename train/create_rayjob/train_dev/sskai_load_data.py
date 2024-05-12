### ray
### ray

def sskai_load_data():
  import pandas as pd
  import torch
  data = pd.read_csv("./salary.csv")
  x = torch.tensor(data["YearsExperience"].values, dtype=torch.float32).reshape(-1, 1)
  y = torch.tensor(data["Salary"].values, dtype=torch.float32).reshape(-1, 1)

  return x, y
