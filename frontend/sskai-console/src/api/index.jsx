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

export const uploadModel = async (user_uid, uid, file) => {
  const res = await axios
    .post(`${url}/upload`, {
      upload_type: 'model',
      user_uid,
      uid,
      filename: file.name
    })
    .catch((err) => err);
  if (!res) {
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
