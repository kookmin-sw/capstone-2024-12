import axios from 'axios';

const url = import.meta.env.VITE_DB_API_URL;

// Model
export const createModel = async (args) => {
  const res = await axios.post(`${url}/models`, args).catch((err) => err);
  return res?.data?.model;
};

export const updateModel = async (uid, args) => {
  const res = await axios.put(`${url}/models/${uid}`, args).catch((err) => err);
  return res?.data?.model;
};

export const getModels = async (user_uid) => {
  const res = await axios
    .get(`${url}/models`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const deleteModel = async (uid) => {
  const res = await axios.delete(`${url}/models/${uid}`);
  return res?.data;
};

// Data
export const getData = async (user_uid) => {
  const res = await axios
    .get(`${url}/data`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

export const createData = async (args) => {
  const res = await axios.post(`${url}/data`, args).catch((err) => err);
  return res?.data?.data;
};

export const updateData = async (uid, args) => {
  const res = await axios.put(`${url}/data/${uid}`, args).catch((err) => err);
  return res?.data?.data;
};

export const deleteData = async (uid) => {
  const res = await axios.delete(`${url}/data/${uid}`).catch((err) => err);
  return res?.data;
};

// Trains
export const getTrains = async (user_uid) => {
  const res = await axios
    .get(`${url}/trains`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

// Inferences
export const getInferences = async (user_uid) => {
  const res = await axios
    .get(`${url}/inferences`, {
      headers: {
        user: user_uid
      }
    })
    .catch((err) => err);
  return res?.data;
};

// Upload Files (Model / Data)
export const uploadS3 = async (upload_type, user_uid, uid, file) => {
  const res = await axios
    .post(`${url}/upload`, {
      upload_type,
      user_uid,
      uid,
      filename: file.name
    })
    .catch((err) => err);

  if (!res?.data) {
    console.log('Pre-signed URL Error');
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
    console.log('Upload Error');
    console.error(res);
    return false;
  }

  return res.data.url.split('?')[0];
};
