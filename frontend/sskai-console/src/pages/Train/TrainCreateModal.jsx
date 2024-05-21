import {
  Alert,
  Button,
  Flex,
  Input,
  InputNumber,
  Modal,
  Select,
  Spin,
  Upload
} from 'antd';
import { ErrorMessage, InputTitle, Title } from '../styles.jsx';
import { InboxOutlined } from '@ant-design/icons';
import { createUserTrain, getData, getModels } from '../../api/index.jsx';
import { useEffect, useState } from 'react';
import { filterObject } from '../../utils/index.jsx';

export default function TrainCreateModal(props) {
  const { isModalOpen, setIsModalOpen, messageApi } = props;

  const [trainName, setTrainName] = useState('');
  const [modelList, setModelList] = useState([]);
  const [modelIdList, setModelIdList] = useState([]);
  const [dataList, setDataList] = useState([]);
  const [dataIdList, setDataIdList] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedData, setSelectedData] = useState(null);
  const [fetchModelLoading, setFetchModelLoading] = useState(false);
  const [fetchDataLoading, setFetchDataLoading] = useState(false);
  const [loading, setLoading] = useState(false);

  // User Model State
  const [epochNum, setEpochNum] = useState(5);
  const [learningRate, setLearningRate] = useState(0.01);
  const [trainSplitSize, setTrainSplitSize] = useState(0.25);
  const [batchSize, setBatchSize] = useState(4);
  const [workerNum, setWorkerNum] = useState(4);
  const [optimStr, setOptimStr] = useState(null);
  const [lossStr, setLossStr] = useState(null);
  const [dataLoader, setDataLoader] = useState([]);

  const uploadSettings = {
    maxCount: 1,
    beforeUpload: (file) => {
      const isZip = ['application/zip'].includes(file.type);
      if (!isZip) {
        messageApi.open({
          type: 'error',
          content: 'Data loader file can only be uploaded as .zip.'
        });
        return Upload.LIST_IGNORE;
      }
      setDataLoader([file]);
      return false;
    },
    fileList: dataLoader,
    onRemove: () => {
      setDataLoader([]);
    }
  };

  const fetchDataList = async () => {
    if (dataList.length) return;
    // TODO: User UID Value Storing in Storage (browser's)
    setFetchDataLoading(true);
    const data = await getData(import.meta.env.VITE_TMP_USER_UID);
    if (!data)
      return messageApi.open({
        type: 'error',
        content: 'Data not found.'
      });
    setDataList(data);
    setDataIdList(
      data.map((d) => ({
        label: `${d.name} (${d.uid})`,
        value: d.uid
      }))
    );
    setFetchDataLoading(false);
  };

  const fetchModelList = async () => {
    if (modelIdList.length) return;
    // TODO: User UID Value Storing in Storage (browser's)
    setFetchModelLoading(true);
    const models = await getModels(import.meta.env.VITE_TMP_USER_UID);
    if (!models)
      return messageApi.open({
        type: 'error',
        content: 'Model not found.'
      });
    setModelList(models);
    setModelIdList(
      models.map((model) => ({
        label: `${model.name} (${model.uid})`,
        value: model.uid
      }))
    );
    setFetchModelLoading(false);
  };

  const handleResetParams = () => {
    setEpochNum(5);
    setLearningRate(0.01);
    setTrainSplitSize(0.25);
    setBatchSize(4);
    setWorkerNum(4);
    setOptimStr(null);
    setLossStr(null);
    setDataLoader(null);
  };

  const handleCancel = () => {
    setTrainName(null);
    setSelectedModel(null);
    setSelectedData(null);
    setModelList([]);
    setModelIdList([]);
    setDataList([]);
    setDataIdList([]);
    handleResetParams();
    setIsModalOpen(false);
  };

  const handleCreateUserTrain = async () => {
    if (
      !trainName ||
      !selectedModel ||
      !selectedData ||
      !optimStr ||
      !lossStr ||
      !dataLoader.length
    )
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });

    setLoading(true);
    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;
    const data = filterObject(selectedData, ['s3_url', 'uid']);
    const model = filterObject(selectedModel, [
      'uid',
      'name',
      'input_shape',
      'value_type',
      'value_range',
      's3_url',
      'max_used_ram',
      'deploy_platform',
      'max_used_gpu_mem',
      'inference_time'
    ]);

    const args = {
      model,
      data,
      user,
      epochNum,
      learningRate,
      trainSplitSize,
      batchSize,
      workerNum,
      optimStr,
      lossStr,
      name: trainName,
      dataLoader: dataLoader[0]
    };

    const newModel = await createUserTrain(args);
    setLoading(false);

    if (!newModel)
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while training the your model. Please try again in a few minutes.'
      });
    messageApi.open({
      type: 'success',
      content: 'The train has been started successfully.'
    });
    handleCancel();
  };

  const handleCreateFMTrain = async () => {};

  const handleChangeModel = (value) => {
    setSelectedModel(modelList.filter(({ uid }) => uid === value).pop());
  };

  const handleChangeData = (value) => {
    setSelectedData(dataList.filter(({ uid }) => uid === value).pop());
  };

  useEffect(() => {
    handleResetParams();
  }, [selectedModel]);

  return (
    <Modal
      title={<Title style={{ fontWeight: 600 }}>New model training</Title>}
      open={isModalOpen}
      onCancel={handleCancel}
      footer={[
        <Button onClick={handleCancel}>Cancel</Button>,
        <Button
          type={'primary'}
          disabled={!selectedModel}
          onClick={
            selectedModel?.type === 'user'
              ? handleCreateUserTrain
              : handleCreateFMTrain
          }
          loading={loading}
        >
          Start
        </Button>
      ]}
    >
      <Flex vertical style={{ marginBottom: '20px' }}>
        <InputTitle>Enter your train name</InputTitle>
        <Alert
          style={{ marginBottom: '10px' }}
          message="A new model will be created with the name you entered."
          type={'info'}
          showIcon
        />
        <Input
          placeholder={'Name'}
          value={trainName}
          onChange={(e) => setTrainName(e.target.value)}
          status={
            trainName && !/^[a-zA-Z0-9-_]{1,20}$/.test(trainName) && 'error'
          }
        />
        {trainName && !/^[a-zA-Z0-9-_]{1,20}$/.test(trainName) && (
          <>
            <ErrorMessage>
              The train name can be up to 20 characters long.
            </ErrorMessage>
            <ErrorMessage>
              Only English and special characters (-, _) can be entered.
            </ErrorMessage>
          </>
        )}
        <InputTitle>Select your model</InputTitle>
        <Select
          showSearch
          notFoundContent={
            fetchModelLoading ? <Spin size={'small'} /> : undefined
          }
          optionFilterProp={'label'}
          placeholder={'Model'}
          onFocus={fetchModelList}
          value={selectedModel?.uid}
          onChange={handleChangeModel}
          options={modelIdList}
        />
        <InputTitle>Select your data</InputTitle>
        <Select
          disabled={!selectedModel}
          showSearch
          notFoundContent={
            fetchDataLoading ? <Spin size={'small'} /> : undefined
          }
          optionFilterProp={'label'}
          placeholder={'Data'}
          onFocus={fetchDataList}
          value={selectedData?.uid}
          onChange={handleChangeData}
          options={dataIdList}
        />

        {/* User Selected User Model (Input Parameters) */}
        {selectedModel?.type === 'user' && (
          <>
            <Flex gap={10}>
              <Flex vertical flex={1}>
                <InputTitle size={'xs'}>Epochs</InputTitle>
                <InputNumber
                  style={{ width: '100%' }}
                  value={epochNum}
                  min={1}
                  max={1000}
                  defaultValue={5}
                  onChange={(value) => setEpochNum(Number(value) || 5)}
                />
              </Flex>
              <Flex vertical flex={1}>
                <InputTitle size={'xs'}>Learning rate</InputTitle>
                <InputNumber
                  style={{ width: '100%' }}
                  value={learningRate}
                  min={0}
                  max={1}
                  defaultValue={0.01}
                  step={0.01}
                  onChange={(value) => setLearningRate(Number(value) || 0.01)}
                />
              </Flex>
              <Flex vertical flex={1}>
                <InputTitle size={'xs'}>Train split size</InputTitle>
                <InputNumber
                  style={{ width: '100%' }}
                  value={trainSplitSize}
                  min={0}
                  max={0.99}
                  defaultValue={0.25}
                  step={0.01}
                  onChange={(value) => setTrainSplitSize(Number(value) || 0.25)}
                />
              </Flex>
            </Flex>
            <Flex gap={10}>
              <Flex vertical flex={1}>
                <InputTitle size={'xs'}>Batch size</InputTitle>
                <InputNumber
                  style={{ width: '100%' }}
                  value={batchSize}
                  min={1}
                  max={64}
                  defaultValue={4}
                  onChange={(value) => setBatchSize(Number(value) || 4)}
                />
              </Flex>
              <Flex vertical flex={1}>
                <InputTitle size={'xs'}>Number of worker(s)</InputTitle>
                <InputNumber
                  style={{ width: '100%' }}
                  value={workerNum}
                  min={1}
                  max={16}
                  defaultValue={4}
                  onChange={(value) => setWorkerNum(Number(value) || 4)}
                />
              </Flex>
            </Flex>
            <InputTitle>Select optimizer</InputTitle>
            <Select
              showSearch
              placeholder={'Optimizer'}
              value={optimStr}
              onChange={(value) => setOptimStr(value)}
              options={[
                { value: 'ASGD', label: 'ASGD' },
                { value: 'Adadelta', label: 'Adadelta' },
                { value: 'Adagrad', label: 'Adagrad' },
                { value: 'Adam', label: 'Adam' },
                { value: 'AdamW', label: 'AdamW' },
                { value: 'Adamax', label: 'Adamax' },
                { value: 'LBFGS', label: 'LBFGS' },
                { value: 'NAdam', label: 'NAdam' },
                { value: 'Optimizer', label: 'Optimizer' },
                { value: 'RAdam', label: 'RAdam' },
                { value: 'RMSprop', label: 'RMSprop' },
                { value: 'Rprop', label: 'Rprop' },
                { value: 'SGD', label: 'SGD' },
                { value: 'SparseAdam', label: 'SparseAdam' }
              ]}
            />
            <InputTitle>Select loss function</InputTitle>
            <Select
              showSearch
              placeholder={'Loss Function'}
              value={lossStr}
              onChange={(value) => setLossStr(value)}
              options={[
                {
                  value: 'AdaptiveLogSoftmaxWithLoss',
                  label: 'AdaptiveLogSoftmaxWithLoss'
                },
                { value: 'BCELoss', label: 'BCELoss' },
                { value: 'BCEWithLogitsLoss', label: 'BCEWithLogitsLoss' },
                { value: 'CTCLoss', label: 'CTCLoss' },
                {
                  value: 'CosineEmbeddingLoss',
                  label: 'CosineEmbeddingLoss'
                },
                { value: 'CrossEntropyLoss', label: 'CrossEntropyLoss' },
                { value: 'GaussianNLLLoss', label: 'GaussianNLLLoss' },
                { value: 'HingeEmbeddingLoss', label: 'HingeEmbeddingLoss' },
                { value: 'HuberLoss', label: 'HuberLoss' },
                { value: 'KLDivLoss', label: 'KLDivLoss' },
                { value: 'L1Loss', label: 'L1Loss' },
                { value: 'MSELoss', label: 'MSELoss' },
                { value: 'MarginRankingLoss', label: 'MarginRankingLoss' },
                {
                  value: 'MultiLabelMarginLoss',
                  label: 'MultiLabelMarginLoss'
                },
                {
                  value: 'MultiLabelSoftMarginLoss',
                  label: 'MultiLabelSoftMarginLoss'
                },
                { value: 'MultiMarginLoss', label: 'MultiMarginLoss' },
                { value: 'NLLLoss', label: 'NLLLoss' },
                { value: 'NLLLoss2d', label: 'NLLLoss2d' },
                { value: 'PoissonNLLLoss', label: 'PoissonNLLLoss' },
                { value: 'SmoothL1Loss', label: 'SmoothL1Loss' },
                { value: 'SoftMarginLoss', label: 'SoftMarginLoss' },
                { value: 'TripletMarginLoss', label: 'TripletMarginLoss' },
                {
                  value: 'TripletMarginWithDistanceLoss',
                  label: 'TripletMarginWithDistanceLoss'
                }
              ]}
            />
            <InputTitle>Upload data loader</InputTitle>
            <Alert
              style={{ marginBottom: '10px' }}
              message={<div style={{ fontWeight: 600 }}>Data Format Guide</div>}
              description={
                <Flex vertical gap={8}>
                  <div>
                    1. There must be only two files in the zip file:
                    sskai_data_load.py and requirements.txt.
                  </div>
                  <div>
                    2. The sskai_data_load.py must include a function named
                    sskai_data_load that returns x and y after preprocessing.
                    The data types of x and y must be torch.tensor.
                  </div>
                  <div>
                    3. Libraries required for preprocessing must be imported
                    within the function.
                  </div>
                </Flex>
              }
              type="info"
              showIcon
            />
            <Upload.Dragger {...uploadSettings}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">
                Click or drag file to this area to upload
              </p>
            </Upload.Dragger>
          </>
        )}

        {/* User Selected Llama Model (Input Parameters) */}
        {selectedModel?.type === 'llama' && (
          <>
            <Flex style={{ marginTop: '10px' }} vertical>
              <Alert
                message={
                  <div style={{ fontWeight: 600 }}>
                    Data Format Guide (for Llama)
                  </div>
                }
                description={
                  <Flex vertical gap={8}>
                    <div>
                      1. The zip file of the data must contain a Parquet format
                      file (.parquet).
                    </div>
                    <div>
                      2. Both ends of the data sentence must be wrapped with
                      &lt;s&gt; text &lt;/s&gt;, and the question must be
                      wrapped with [INST] question [/INST].
                    </div>
                    <div>
                      (Example) &lt;s&gt; [INST] question [/INST] answer
                      &lt;/s&gt;
                    </div>
                  </Flex>
                }
                type="info"
                showIcon
              />
              <InputTitle>Epochs</InputTitle>
              <InputNumber
                style={{ width: '100%' }}
                value={epochNum}
                min={1}
                max={1000}
                defaultValue={5}
                onChange={(value) => setEpochNum(Number(value) || 5)}
              />
            </Flex>
          </>
        )}

        {/* User Selected Diffusion Model (Input Parameters) */}
        {selectedModel?.type === 'diffusion' && (
          <>
            <Flex style={{ marginTop: '10px' }} vertical>
              <Alert
                message={
                  <div style={{ fontWeight: 600 }}>
                    Data Format Guide (for Diffusion)
                  </div>
                }
                description={
                  <Flex vertical gap={8}>
                    <div>
                      1. Images should have a .jpg extension, a 1:1 aspect
                      ratio, and no size limit; however, larger images will
                      result in higher quality.
                    </div>
                    <div>
                      2. The compressed data should be organized into
                      /class_data (200 images) and /user_data (5 images)
                      directories.
                    </div>
                    <div>
                      3. The class_data should contain approximately 40 times
                      more images than user_data.
                    </div>
                  </Flex>
                }
                type="info"
                showIcon
              />
              <InputTitle>Enter your class</InputTitle>
              <Alert
                type="warning"
                showIcon
                message={
                  'You must specify the class during training. Dog (O) / Animal (X)'
                }
                style={{ marginBottom: '10px' }}
              />
              <Input placeholder={'Class'} />
              <InputTitle size={'xs'}>Epochs</InputTitle>
              <InputNumber
                style={{ width: '100%' }}
                value={epochNum}
                min={1}
                max={1000}
                defaultValue={5}
                onChange={(value) => setEpochNum(Number(value) || 5)}
              />
            </Flex>
          </>
        )}
      </Flex>
    </Modal>
  );
}
