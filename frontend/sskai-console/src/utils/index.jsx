export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const seconds = date.getSeconds().toString().padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
};

export const calculateDuration = (startTime, endTime) => {
  const duration = endTime - startTime;
  const seconds = Math.floor((duration / 1000) % 60);
  const minutes = Math.floor((duration / 1000 / 60) % 60);
  const hours = Math.floor(duration / 1000 / 60 / 60);

  const paddedHours = hours.toString().padStart(2, '0');
  const paddedMinutes = minutes.toString().padStart(2, '0');
  const paddedSeconds = seconds.toString().padStart(2, '0');

  return isNaN(paddedSeconds)
    ? '00:00:00'
    : `${paddedHours}:${paddedMinutes}:${paddedSeconds}`;
};

export const copyToClipBoard = (text) => {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
};

export const filterObject = (obj, keys) => {
  return keys.reduce((result, key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
    return result;
  }, {});
};

export const calculateCost = (startTime, endTime, cost) => {
  console.log(startTime, endTime);
  const elapsedTimeSeconds = Math.floor((endTime - startTime) / 1000);
  console.log(Math.floor(elapsedTimeSeconds * cost * 10000));
  return Math.floor(elapsedTimeSeconds * cost * 10000) / 10000;
};
