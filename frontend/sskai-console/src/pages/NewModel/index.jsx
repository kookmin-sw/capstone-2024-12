import { ErrorMessage, InputTitle, PageLayout, Title } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import Icon, { InboxOutlined, CheckCircleTwoTone } from '@ant-design/icons';
import {
  Button,
  Flex,
  Input,
  message,
  Radio,
  Select,
  Steps,
  Upload
} from 'antd';
import { useEffect, useState } from 'react';
import {
  createModel,
  deleteModel,
  getModel,
  profileModel,
  updateModel,
  uploadS3,
  uploadS3Multipart
} from '../../api/index.jsx';
import { useNavigate } from 'react-router-dom';
import IconRadio from '../../components/IconRadio/index.jsx';
import MetaSVG from '../../assets/meta-logo.svg?react';
import StableSVG from '../../assets/stability-ai-logo.svg?react';
import OpenAISVG from '../../assets/open-ai-logo.svg?react';
import GoogleSVG from '../../assets/google-logo.svg?react';
import ClaudeSVG from '../../assets/claude-logo.svg?react';

const STEP_ITEM = [
  {
    title: 'Basic Information'
  },
  {
    title: 'Configuration'
  },
  {
    title: 'Finished'
  }
];

const MetaLogo = (props) => <Icon component={MetaSVG} {...props} />;
const StableLogo = (props) => <Icon component={StableSVG} {...props} />;
const OpenAILogo = (props) => <Icon component={OpenAISVG} {...props} />;
const GoogleLogo = (props) => <Icon component={GoogleSVG} {...props} />;
const ClaudeLogo = (props) => <Icon component={ClaudeSVG} {...props} />;

export default function NewModel(props) {
  const navigate = useNavigate();

  const [messageApi, contextHolder] = message.useMessage();
  const [stepPos, setStepPos] = useState(0);
  const [modelName, setModelName] = useState('');
  const [modelType, setModelType] = useState('');
  const [modelFile, setModelFile] = useState([]);
  const [loading, setLoading] = useState(false);
  const [inputShape, setInputShape] = useState('');
  const [valueType, setValueType] = useState('float32');
  const [valueRange, setValueRange] = useState('');
  const [fmType, setFmType] = useState('llama');
  const [fetchFM, setFetchFM] = useState([]);

  const uploadSettings = {
    maxCount: 1,
    beforeUpload: (file) => {
      const isZip = ['application/zip'].includes(file.type);
      if (!isZip) {
        messageApi.open({
          type: 'error',
          content: 'Model file can only be uploaded as .zip.'
        });
        return Upload.LIST_IGNORE;
      }
      setModelFile([file]);
      return false;
    },
    fileList: modelFile,
    onRemove: () => {
      setModelFile([]);
    }
  };

  const handleFetchFM = async () => {
    const fms = [];
    fms.push(await getModel('llama'));
    fms.push(await getModel('diffusion'));
    setFetchFM(fms);
  };

  const handleBasicInfoSubmit = () => {
    if (modelName && /^[a-zA-Z0-9-_]{1,20}$/.test(modelName) && modelType)
      setStepPos(1);
    else
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });
  };

  const handleCreateUserModel = async () => {
    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;

    if (
      !inputShape ||
      !valueType ||
      !/^\(-?\d+(?:,\s*-?\d+)*\)$/.test(inputShape) ||
      (valueRange && !/^\(-?\d+,\s*-?\d+\)$/.test(valueRange))
    )
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });

    if (!modelFile.length)
      return messageApi.open({
        type: 'error',
        content: 'Please upload your model file.'
      });

    setLoading(true);

    const model = await createModel({
      user,
      name: modelName,
      type: modelType,
      input_shape: inputShape,
      value_type: valueType,
      value_range: valueRange
    });

    if (!model) {
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while creating the your model. Please try again in a few minutes.'
      });
    }

    const uploaded =
      modelFile[0].size < 5 * 1024 * 1024 * 1024 // 5GB
        ? await uploadS3('model', user, modelFile.uid, modelFile[0])
        : await uploadS3Multipart('model', user, modelFile.uid, modelFile[0]);

    if (!uploaded) {
      await deleteModel(model.uid);
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while uploading the your model. Please try again in a few minutes.'
      });
    }

    const isUpdated = await updateModel(model.uid, {
      s3_url: uploaded
    });

    if (!isUpdated) {
      await deleteModel(model.uid);
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while updating the your model. Please try again in a few minutes.'
      });
    }

    await profileModel(model.uid);

    setLoading(false);
    setStepPos(2);
  };

  const handleCreateFMModel = async () => {
    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;

    if (!fmType)
      return messageApi.open({
        type: 'error',
        content: 'Please select Foundation Model.'
      });

    const selectedFM = fetchFM.filter(({ uid }) => uid === fmType).pop();

    if (!selectedFM)
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while creating the your model. Please try again in a few minutes.'
      });

    setLoading(true);

    const model = await createModel({
      ...selectedFM,
      user,
      name: modelName,
      type: fmType
    });

    setLoading(false);

    if (!model)
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while creating the your model. Please try again in a few minutes.'
      });

    setStepPos(2);
  };

  useEffect(() => {
    if (stepPos === 1 && modelType === 'fm' && !fetchFM.length) handleFetchFM();
  }, [modelType, stepPos]);
  return (
    <PageLayout>
      {contextHolder}
      <Section>
        <Flex
          style={{
            width: '600px',
            alignSelf: 'center'
          }}
          vertical
          gap={30}
        >
          <Steps
            size={'small'}
            current={stepPos}
            items={STEP_ITEM}
            responsive={false}
          />
          {stepPos === 0 && (
            <>
              <Flex vertical gap={10}>
                <Flex vertical gap={5}>
                  <InputTitle>Enter your model name</InputTitle>
                  <Input
                    placeholder={'Name'}
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    status={
                      modelName &&
                      !/^[a-zA-Z0-9-_]{1,20}$/.test(modelName) &&
                      'error'
                    }
                  />
                  {modelName && !/^[a-zA-Z0-9-_]{1,20}$/.test(modelName) && (
                    <>
                      <ErrorMessage>
                        The model name can be up to 20 characters long.
                      </ErrorMessage>
                      <ErrorMessage>
                        Only English and special characters (-, _) can be
                        entered.
                      </ErrorMessage>
                    </>
                  )}
                </Flex>
                <Flex vertical gap={5}>
                  <InputTitle>Select your model type</InputTitle>
                  <Radio.Group
                    buttonStyle={'solid'}
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                  >
                    <Radio.Button value={'user'}>
                      Own model (Upload)
                    </Radio.Button>
                    <Radio.Button value={'fm'}>Foundation Model</Radio.Button>
                  </Radio.Group>
                </Flex>
              </Flex>
              <Flex style={{ alignSelf: 'end', marginBottom: '20px' }}>
                <Button type={'primary'} onClick={handleBasicInfoSubmit}>
                  Next
                </Button>
              </Flex>
            </>
          )}
          {stepPos === 1 && modelType === 'user' && (
            <>
              <Flex vertical gap={5}>
                <InputTitle>Input shape of your model</InputTitle>
                <Input
                  value={inputShape}
                  onChange={(e) => setInputShape(e.target.value)}
                  onBlur={() => setInputShape((str) => str.replace(/ /gi, ''))}
                  placeholder={
                    '(batchsize, width), (batchsize, channel, height, width) or ...'
                  }
                  status={
                    inputShape &&
                    !/^\(-?\d+(?:,\s*-?\d+)*\)$/.test(
                      inputShape.replace(/ /gi, '')
                    ) &&
                    'error'
                  }
                />
                {inputShape &&
                  !/^\(-?\d+(?:,\s*-?\d+)*\)$/.test(
                    inputShape.replace(/ /gi, '')
                  ) && (
                    <>
                      <ErrorMessage>
                        The shape of the model must be written in the correct
                        format
                      </ErrorMessage>
                      <ErrorMessage>
                        with numbers in parentheses separated by commas.
                      </ErrorMessage>
                    </>
                  )}
                <InputTitle>Input type of your model</InputTitle>
                <Select
                  showSearch
                  labelInValue
                  optionFilterProp={'label'}
                  placeholder={'Type'}
                  onChange={({ value }) => setValueType(value)}
                  defaultValue={'float32'}
                  options={[
                    { value: 'float16', label: 'float16' },
                    { value: 'float32', label: 'float32 (default)' },
                    { value: 'float64', label: 'float64' },
                    { value: 'int8', label: 'int8' },
                    { value: 'int16', label: 'int16' },
                    { value: 'int32', label: 'int32' },
                    { value: 'int64', label: 'int64' },
                    { value: 'bool', label: 'bool' }
                  ]}
                ></Select>
                <InputTitle>Input range of your model (optional)</InputTitle>
                <Input
                  placeholder={'(min, max)'}
                  value={valueRange}
                  onChange={(e) => setValueRange(e.target.value)}
                  onBlur={() => setValueRange((str) => str.replace(/ /gi, ''))}
                  status={
                    valueRange &&
                    !/^\(-?\d+,\s*-?\d+\)$/.test(
                      valueRange.replace(/ /gi, '')
                    ) &&
                    'error'
                  }
                />
                {valueRange &&
                  !/^\(-?\d+,\s*-?\d+\)$/.test(
                    valueRange.replace(/ /gi, '')
                  ) && (
                    <>
                      <ErrorMessage>
                        The model's range must be written in parentheses as two
                        numbers separated by a comma.
                      </ErrorMessage>
                    </>
                  )}

                <InputTitle>Upload your model</InputTitle>
                <Upload.Dragger {...uploadSettings} disabled={loading}>
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                  </p>
                  <p className="ant-upload-text">
                    Click or drag file to this area to upload
                  </p>
                </Upload.Dragger>
              </Flex>
            </>
          )}
          {stepPos === 1 && modelType === 'fm' && (
            <>
              <Flex vertical>
                <InputTitle>Select Foundation Model (FM)</InputTitle>
                <IconRadio
                  selected={fmType}
                  setSelected={setFmType}
                  loading={!fetchFM?.length}
                  options={[
                    {
                      key: 'llama',
                      label: 'llama-2-7b-chat-hf',
                      icon: MetaLogo
                    },
                    {
                      key: 'diffusion',
                      label: 'stable-diffusion-v1-4',
                      icon: StableLogo
                    },
                    {
                      key: 'gpt',
                      label: 'gpt-2',
                      icon: OpenAILogo,
                      disabled: true
                    },
                    {
                      key: 'dalle',
                      label: 'dalle-3',
                      icon: OpenAILogo,
                      disabled: true
                    },
                    {
                      key: 'bert',
                      label: 'bert',
                      icon: GoogleLogo,
                      disabled: true
                    },
                    {
                      key: 'claude',
                      label: 'claude2',
                      icon: ClaudeLogo,
                      disabled: true
                    }
                  ]}
                />
              </Flex>
            </>
          )}
          {stepPos === 1 && (
            <>
              <Flex style={{ alignSelf: 'end', margin: '20px 0' }} gap={10}>
                <Button
                  type={'default'}
                  onClick={() => setStepPos(0)}
                  disabled={loading}
                >
                  Previous
                </Button>
                <Button
                  type={'primary'}
                  onClick={
                    modelType === 'user'
                      ? handleCreateUserModel
                      : handleCreateFMModel
                  }
                  loading={loading}
                >
                  Next
                </Button>
              </Flex>
            </>
          )}
          {stepPos === 2 && (
            <>
              <Flex align={'center'} justify={'center'} gap={20}>
                <CheckCircleTwoTone style={{ fontSize: '40px' }} />
                <Title>The model was created successfully.</Title>
              </Flex>
              <Flex style={{ alignSelf: 'end', marginBottom: '20px' }} gap={10}>
                <Button
                  type={'default'}
                  onClick={() => {
                    navigate('/dashboard');
                  }}
                >
                  Dashboard
                </Button>
                <Button
                  type={'primary'}
                  onClick={() => {
                    navigate('/model');
                  }}
                >
                  Show Model
                </Button>
              </Flex>
            </>
          )}
        </Flex>
      </Section>
    </PageLayout>
  );
}
