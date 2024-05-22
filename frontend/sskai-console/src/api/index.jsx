import axios from 'axios';

const DB_API = import.meta.env.VITE_DB_API_URL;
const INFERENCE_SPOT_API = import.meta.env.VITE_INFERENCE_SPOT_API_URL;
const INFERENCE_SERVERLESS_API = import.meta.env
  .VITE_INFERENCE_SERVERLESS_API_URL;
const MODEL_PROFILE_API = import.meta.env.VITE_MODEL_PROFILE_API_URL;
const USER_TRAIN_API = import.meta.env.VITE_USER_TRAIN_API_URL;
const LLAMA_TRAIN_API = import.meta.env.VITE_LLAMA_TRAIN_API_URL;
const DIFFUSION_TRAIN_API = import.meta.env.VITE_DIFFUSION_TRAIN_API_URL;
const STREAMLIT_API = import.meta.env.VITE_STREAMLIT_API_URL;
const INFERENCE_LLAMA_API = import.meta.env.VITE_INFERENCE_LLAMA_API_URL;
const INFERENCE_DIFFUSION_API = import.meta.env
  .VITE_INFERENCE_DIFFUSION_API_URL;

// Model
export const createModel = async (args) => {
  const res = await axios.post(`${DB_API}/models`, args).catch((err) => err);
  if (res?.data)
    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'model',
      job: 'Model Created'
    });
  return res?.data?.model;
};

export const updateModel = async (uid, args) => {
  const res = await axios
    .put(`${DB_API}/models/${uid}`, args)
    .catch((err) => err);
  if (res?.data && args?.name) {
    const { model } = res.data;
    await createLog({
      user: model.user,
      name: model.name,
      kind_of_job: 'model',
      job: 'Model Updated'
    });
  }
  return res?.data?.model;
};

export const getModel = async (uid) => {
  const res = await axios.get(`${DB_API}/models/${uid}`).catch((err) => err);
  return res?.data;
};

export const getModels = async (user_uid) => {
  const res = await axios
    .get(`${DB_API}/models`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const deleteModel = async (uid) => {
  const res = await axios.delete(`${DB_API}/models/${uid}`).catch((err) => err);
  if (res?.data) {
    const { model } = res.data;
    await createLog({
      user: model.user,
      name: model.name,
      kind_of_job: 'model',
      job: 'Model Deleted'
    });
  }
  return res?.data;
};

export const profileModel = async (uid) => {
  const res = await axios
    .post(`${MODEL_PROFILE_API}`, {
      uid
    })
    .catch((err) => err);
};

// Data
export const getData = async (user_uid) => {
  const res = await axios
    .get(`${DB_API}/data`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const createData = async (args) => {
  const res = await axios.post(`${DB_API}/data`, args).catch((err) => err);
  if (res?.data)
    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'data',
      job: 'Data Created'
    });
  return res?.data?.data;
};

export const updateData = async (uid, args) => {
  const res = await axios
    .put(`${DB_API}/data/${uid}`, args)
    .catch((err) => err);
  if (res?.data && args?.name) {
    const { data } = res.data;
    await createLog({
      user: data.user,
      name: data.name,
      kind_of_job: 'data',
      job: 'Data Updated'
    });
  }
  return res?.data?.data;
};

export const deleteData = async (uid) => {
  const res = await axios.delete(`${DB_API}/data/${uid}`).catch((err) => err);
  if (res?.data) {
    const { data } = res.data;
    await createLog({
      user: data.user,
      name: data.name,
      kind_of_job: 'data',
      job: 'Data Deleted'
    });
  }
  return res?.data;
};

// Trains
export const getTrains = async (user_uid) => {
  const res = await axios
    .get(`${DB_API}/trains`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const createUserTrain = async (args) => {
  const res = await axios
    .post(`${DB_API}/trains`, {
      user: args.user,
      name: args.name,
      model: args.model.uid,
      data: args.data.uid,
      epoch_num: args.epochNum,
      learning_rate: args.learningRate,
      optim_str: args.optimStr,
      loss_str: args.lossStr,
      train_split_size: args.trainSplitSize,
      batch_size: args.batchSize,
      worker_num: args.workerNum,
      model_type: args.model.type
    })
    .catch((err) => err);

  if (!res?.data) {
    console.log(res);
    return false;
  }

  const { train } = res.data;

  const uploaded = await uploadS3(
    'data_loader',
    args.user,
    train.uid,
    args.dataLoader
  );

  if (!uploaded) return false;

  await axios
    .put(`${DB_API}/trains/${train.uid}`, {
      data_loader_path: uploaded
    })
    .catch((err) => err);

  const model = await createModel({
    user: args.user,
    name: args.name,
    type: 'user',
    input_shape: args.model.input_shape,
    value_type: args.model.value_type,
    value_range: args.model.value_range,
    deploy_platform: args.model.deploy_platform,
    max_used_ram: args.model.max_used_ram,
    max_used_gpu_ram: args.model.max_used_gpu_ram,
    inference_time: args.model.inference_time
  }).catch((err) => err);

  if (!model) return false;

  await axios
    .post(USER_TRAIN_API, {
      epoch_num: args.epochNum,
      learning_rate: args.learningRate,
      train_split_size: args.trainSplitSize,
      batch_size: args.batchSize,
      worker_num: args.workerNum,
      optim_str: args.optimStr,
      loss_str: args.lossStr,
      uid: train.uid,
      user_uid: args.user,
      model_uid: model.uid,
      model_s3_url: args.model.s3_url,
      data_s3_url: args.data.s3_url,
      data_load_s3_url: uploaded,
      ram_size: args.model.max_used_ram,
      action: 'create'
    })
    .catch((err) => err);

  await createCost('train', train.uid, 'nodepool-1', 0);

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'train',
    job: 'Train Created'
  });

  return model;
};

export const createDiffusionTrain = async (args) => {
  const res = await axios
    .post(`${DB_API}/trains`, {
      user: args.user,
      name: args.name,
      model: args.model.uid,
      data: args.data.uid,
      epoch_num: args.epochNum,
      class: args.dataClass,
      model_type: 'diffusion'
    })
    .catch((err) => err);

  if (!res?.data) {
    console.log(res);
    return false;
  }

  const { train } = res.data;

  const model = await createModel({
    user: args.user,
    name: args.name,
    type: 'diffusion',
    deploy_platform: 'nodepool-2'
  }).catch((err) => err);

  if (!model) return false;

  await axios
    .post(DIFFUSION_TRAIN_API, {
      action: 'create',
      uid: train.uid,
      user_uid: args.user,
      model_uid: model.uid,
      model_s3_url: args.model.s3_url,
      data_s3_url: args.data.s3_url,
      epoch_num: args.epochNum,
      data_class: args.dataClass
    })
    .catch((err) => err);

  await createCost('train', train.uid, 'nodepool-2', 0);

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'train',
    job: 'Train Created'
  });

  return model;
};

export const createLlamaTrain = async (args) => {
  const res = await axios
    .post(`${DB_API}/trains`, {
      user: args.user,
      name: args.name,
      model: args.model.uid,
      data: args.data.uid,
      epoch_num: args.epochNum,
      model_type: 'llama'
    })
    .catch((err) => err);

  if (!res?.data) {
    console.log(res);
    return false;
  }

  const { train } = res.data;

  const model = await createModel({
    user: args.user,
    name: args.name,
    type: 'llama',
    deploy_platform: 'nodepool-2'
  }).catch((err) => err);

  if (!model) return false;

  await axios
    .post(LLAMA_TRAIN_API, {
      action: 'create',
      uid: train.uid,
      user_uid: args.user,
      model_uid: model.uid,
      model_s3_url: args.model.s3_url,
      data_s3_url: args.data.s3_url,
      epoch: args.epochNum
    })
    .catch((err) => err);

  await createCost('train', train.uid, 'nodepool-2', 0);

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'train',
    job: 'Train Created'
  });

  return model;
};

export const deleteTrain = async (uid, type, status, user, name) => {
  if (status !== 'Completed') {
    if (type === 'diffusion') await stopDiffusionTrain(uid);
    else if (type === 'llama') await stopLlamaTrain(uid);
    else await stopUserTrain(uid);
  }

  await axios.delete(`${DB_API}/trains/${uid}`);

  await createLog({
    user,
    name,
    kind_of_job: 'train',
    job: 'Train Deleted'
  });
};

const stopUserTrain = async (uid) => {
  await axios
    .post(USER_TRAIN_API, {
      action: 'delete',
      uid
    })
    .catch((err) => err);
};

const stopDiffusionTrain = async (uid) => {
  await axios
    .post(DIFFUSION_TRAIN_API, {
      action: 'delete',
      uid
    })
    .catch((err) => err);
};

const stopLlamaTrain = async (uid) => {
  await axios
    .post(LLAMA_TRAIN_API, {
      action: 'delete',
      uid
    })
    .catch((err) => err);
};

// Inferences
export const createSpotInference = async (args) => {
  const res = await axios
    .post(`${DB_API}/inferences`, {
      user: args.user,
      name: args.name,
      model: args.model,
      model_type: args.model_type,
      type: args.type
    })
    .catch((err) => err);

  if (!res?.data) {
    console.error(res);
    return false;
  }

  const { Item } = res.data.inference;

  const spot = await axios
    .post(`${INFERENCE_SPOT_API}`, {
      uid: Item.uid,
      user: args.user,
      action: 'create',
      model: args.model_detail
    })
    .catch((err) => err);

  if (spot.status !== 200) {
    await axios.delete(`${DB_API}/inferences/${Item.uid}`);
    return false;
  }

  await createCost(
    'inference',
    Item.uid,
    args.model_detail.deployment_type,
    args.model_detail.max_used_ram
  );

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'inference',
    job: 'Endpoint (using Spot) Created'
  });

  return Item;
};

export const deleteSpotInference = async (args) => {
  const spot = await axios
    .post(`${INFERENCE_SPOT_API}`, {
      uid: args.uid,
      user: args.user,
      action: 'delete'
    })
    .catch((err) => err);

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'inference',
    job: 'Endpoint (using Spot) Deleted'
  });

  return spot.status === 200;
};

export const updateInference = async (uid, args) => {
  const res = await axios
    .put(`${DB_API}/inferences/${uid}`, args)
    .catch((err) => err);
  if (res?.data) {
    const { inference } = res.data;
    await createLog({
      user: inference.user,
      name: inference.name,
      kind_of_job: 'inference',
      job: 'Endpoint Updated'
    });
  }
  return res?.data;
};

export const createServerlessInference = async (args) => {
  const res = await axios
    .post(`${DB_API}/inferences`, {
      user: args.user,
      name: args.name,
      model: args.model,
      model_type: args.model_type,
      type: args.type
    })
    .catch((err) => err);

  if (!res?.data) {
    console.error(res);
    return false;
  }

  const { Item } = res.data.inference;

  const serverless = await axios
    .post(`${INFERENCE_SERVERLESS_API}`, {
      uid: Item.uid,
      user: args.user,
      action: 'create',
      model: args.model_detail
    })
    .catch((err) => err);

  if (serverless.status !== 200) {
    await axios.delete(`${DB_API}/inferences/${Item.uid}`);
    return false;
  }

  await createCost(
    'inference',
    Item.uid,
    'Serverless',
    Number(args.model_detail.max_used_ram)
  );

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'inference',
    job: 'Endpoint (using Serverless) Created'
  });

  return Item;
};

export const deleteServerlessInference = async (args) => {
  const model = await getModel(args.model);
  if (!model) return false;
  const serverless = await axios
    .post(`${INFERENCE_SERVERLESS_API}`, {
      uid: args.uid,
      user: args.user,
      action: 'delete',
      model: {
        s3_url: model.s3_url,
        max_used_ram: model.max_used_ram || 5120
      }
    })
    .catch((err) => err);

  await createLog({
    user: args.user,
    name: args.name,
    kind_of_job: 'inference',
    job: 'Endpoint (using Serverless) Deleted'
  });

  return serverless.status === 200;
};

export const getInferences = async (user_uid) => {
  const res = await axios
    .get(`${DB_API}/inferences`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const manageStreamlit = async ({
  user,
  uid,
  model_type,
  endpoint_url,
  action,
  name
}) => {
  const res = await axios
    .post(STREAMLIT_API, {
      user,
      uid,
      action,
      model_type,
      endpoint_url
    })
    .catch((err) => err);

  await createLog({
    user,
    name,
    kind_of_job: 'inference',
    job: `Testbed ${action === 'create' ? 'Deployed' : 'Un-Deployed'}`
  });

  return res.status === 200;
};

export const createFMInference = async (type, args) => {
  const res = await axios
    .post(`${DB_API}/inferences`, {
      user: args.user,
      name: args.name,
      model: args.model,
      model_type: args.model_type,
      type: args.type
    })
    .catch((err) => err);

  if (!res?.data) {
    console.error(res);
    return false;
  }

  const { Item } = res.data.inference;

  if (type === 'llama') {
    const llama = await axios
      .post(`${INFERENCE_LLAMA_API}`, {
        uid: Item.uid,
        user: args.user,
        action: 'create',
        model: args.model_detail
      })
      .catch((err) => err);

    if (llama.status !== 200) {
      await axios.delete(`${DB_API}/inferences/${Item.uid}`);
      return false;
    }

    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'inference',
      job: 'Endpoint (Llama) Created'
    });
  } else if (type === 'diffusion') {
    const diffusion = await axios
      .post(`${INFERENCE_DIFFUSION_API}`, {
        uid: Item.uid,
        user: args.user,
        action: 'create',
        model: args.model_detail
      })
      .catch((err) => err);

    if (diffusion.status !== 200) {
      await axios.delete(`${DB_API}/inferences/${Item.uid}`);
      return false;
    }

    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'inference',
      job: 'Endpoint (Diffusion) Created'
    });
  }

  await createCost('inference', Item.uid, 'nodepool-2', 0);
  return Item;
};

export const deleteFMInference = async (type, args) => {
  if (type === 'llama') {
    const llama = await axios
      .post(`${INFERENCE_LLAMA_API}`, {
        uid: args.uid,
        user: args.user,
        action: 'delete'
      })
      .catch((err) => err);

    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'inference',
      job: 'Endpoint (Llama) Deleted'
    });

    return llama.status === 200;
  } else if (type === 'diffusion') {
    const diffusion = await axios
      .post(`${INFERENCE_DIFFUSION_API}`, {
        uid: args.uid,
        user: args.user,
        action: 'delete'
      })
      .catch((err) => err);

    await createLog({
      user: args.user,
      name: args.name,
      kind_of_job: 'inference',
      job: 'Endpoint (Diffusion) Deleted'
    });

    return diffusion.status === 200;
  }
};

// Upload Files (Model / Data)
export const uploadS3 = async (upload_type, user_uid, uid, file) => {
  const res = await axios
    .post(`${DB_API}/upload`, {
      upload_type,
      user_uid,
      uid,
      filename: file.name
    })
    .catch((err) => err);

  if (!res?.data) {
    console.error(res);
    return false;
  }
  const upload = await axios
    .put(res.data.url, file, {
      headers: {
        'Content-Type': file.type
      }
    })
    .catch((err) => err);
  if (!upload || upload.status !== 200) {
    console.error(res);
    return false;
  }

  return res.data.url.split('?')[0];
};

export const uploadS3Multipart = async (upload_type, user_uid, uid, file) => {
  const args = {
    upload_type,
    user_uid,
    uid,
    filename: file.name
  };
  const res = await axios
    .post(`${DB_API}/upload/start`, args)
    .catch((err) => err);

  const { UploadId } = res?.data;

  if (!UploadId) {
    console.error(res);
    return false;
  }

  const CHUNK_SIZE = 5 * 1024 * 1024 * 1024; // 5GB
  const totalChunk = Math.floor(file.size / CHUNK_SIZE) + 1;
  const chunkPromises = [];

  for (let index = 1; index <= totalChunk; index++) {
    const start = (index - 1) * CHUNK_SIZE;
    const end = index * CHUNK_SIZE;
    const chunk =
      index < totalChunk ? file.slice(start, end) : file.slice(start);

    const presigned = await axios.post(`${DB_API}/upload/url`, {
      ...args,
      UploadId,
      PartNumber: index
    });

    const url = presigned.data;

    if (!url) {
      console.error(url);
      return false;
    }

    const upload = axios.put(url, chunk, {
      headers: {
        'Content-type': file.type
      }
    });
    chunkPromises.push(upload);
  }

  const resolved = await Promise.all(chunkPromises);
  const Parts = [];

  resolved.forEach((promise, index) => {
    Parts.push({
      ETag: promise.headers.etag,
      PartNumber: index + 1
    });
  });

  const completed = await axios.post(`${DB_API}/upload/complete`, {
    ...args,
    UploadId,
    Parts
  });

  if (completed.status !== 200) {
    console.err(completed);
    return false;
  }

  return true;
};

// Logs
export const getLogs = async (user_uid) => {
  const res = await axios
    .get(`${DB_API}/logs`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

const createLog = async ({ user, name, kind_of_job, job }) => {
  await axios
    .post(`${DB_API}/logs`, {
      user,
      name,
      kind_of_job,
      job
    })
    .catch((err) => err);
};

// Cost

const createCost = async (type, uid, platform, ram) => {
  await axios
    .post(`${DB_API}/cost`, {
      type,
      uid,
      platform,
      ram
    })
    .catch((err) => err);
};
