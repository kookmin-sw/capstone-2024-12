import { PageLayout } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import { InboxOutlined, CheckCircleTwoTone } from '@ant-design/icons';
import { Button, Flex, Input, message, Radio, Steps, Upload } from 'antd';
import { useState } from 'react';
import styled from 'styled-components';
import { createModel, updateModel, uploadModel } from '../../api/index.jsx';
import { useNavigate } from 'react-router-dom';

const ErrorMessage = styled.div`
  color: #ff4d4f;
`;

const Title = styled.div`
  font-size: 18px;
  font-weight: 500;
`;

const InputTitle = styled(Title)`
  margin-top: 12px;
  margin-bottom: 4px;
`;

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

export default function NewModel(props) {
  const navigate = useNavigate();

  const [messageApi, contextHolder] = message.useMessage();
  const [stepPos, setStepPos] = useState(0);
  const [modelName, setModelName] = useState('');
  const [modelType, setModelType] = useState('');
  const [modelFile, setModelFile] = useState([]);
  const [loading, setLoading] = useState(false);
  const [createdModel, setCreatedModel] = useState(null);

  const uploadSettings = {
    maxCount: 1,
    beforeUpload: (file) => {
      const isModel = ['application/zip', 'application/tar+gzip'].includes(
        file.type
      );
      if (!isModel) {
        messageApi.open({
          type: 'error',
          content: 'Model files can only be uploaded as .zip or .tar.gz.'
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

  const handleCreateModel = async () => {
    const user = '5d9b890e-1316-4e25-8f67-829702a24331';

    if (!modelFile.length)
      return messageApi.open({
        type: 'error',
        content: 'Please upload your model file.'
      });

    setLoading(true);

    const model =
      createdModel ||
      (await createModel({
        user,
        name: modelName,
        type: modelType
      }));

    if (!model) {
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while creating the your model. Please try again in a few minutes.'
      });
    }

    if (!createdModel) setCreatedModel(model);

    const uploaded = await uploadModel(user, model.uid, modelFile[0]);

    if (!uploaded) {
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
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while updating the your model. Please try again in a few minutes.'
      });
    }

    setCreatedModel(null);
    setLoading(false);
    setStepPos(2);
  };

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
          <Steps size={'small'} current={stepPos} items={STEP_ITEM} />
          {stepPos === 0 && (
            <>
              <Flex vertical gap={10}>
                <Flex vertical gap={5}>
                  <InputTitle>Input your model name</InputTitle>
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
                    <Radio.Button value={'fm'} disabled>
                      Foundation Model
                    </Radio.Button>
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
              <Flex vertical>
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
              <div>fm</div>
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
                  onClick={handleCreateModel}
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
                    navigate('/models');
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
