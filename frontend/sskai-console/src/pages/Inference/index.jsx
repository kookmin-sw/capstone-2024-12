import {
  ErrorMessage,
  InputTitle,
  PageLayout,
  TableToolbox,
  Title
} from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import {
  Button,
  Dropdown,
  Flex,
  Input,
  message,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Table,
  Tag
} from 'antd';
import { useEffect, useState } from 'react';
import {
  createFMInference,
  createServerlessInference,
  createSpotInference,
  deleteFMInference,
  deleteServerlessInference,
  deleteSpotInference,
  getInferences,
  getModel,
  getModels,
  manageStreamlit,
  updateInference
} from '../../api/index.jsx';
import {
  PlusOutlined,
  SearchOutlined,
  SyncOutlined,
  CopyOutlined,
  QuestionCircleOutlined,
  CloudOutlined,
  CloudServerOutlined,
  LinkOutlined
} from '@ant-design/icons';
import axios from 'axios';
import IconRadio from '../../components/IconRadio/index.jsx';
import { calculateCost, copyToClipBoard } from '../../utils/index.jsx';
import { useNavigate } from 'react-router-dom';

const INFERENCE_TABLE_COLUMNS = [
  {
    title: 'Name',
    dataIndex: 'name',
    key: 'name',
    width: 150
  },
  {
    title: 'Type',
    dataIndex: 'type',
    width: 120,
    render: (type) => {
      const color = type === 'Spot' ? 'green' : 'geekblue';
      return (
        <Tag color={color} key={type}>
          {type.toUpperCase()}
        </Tag>
      );
    }
  },
  {
    title: 'Endpoint URL',
    dataIndex: 'endpoint',
    key: 'endpoint',
    width: 200,
    ellipsis: true,
    render: (url) => (
      <Flex justify={'space-between'}>
        <div
          style={{
            width: '90%',
            textOverflow: 'ellipsis',
            overflow: 'hidden',
            whiteSpace: 'nowrap'
          }}
        >
          {url}
        </div>
        <CopyOutlined
          style={{ cursor: 'pointer', color: 'rgba(0, 0, 0, 0.45)' }}
          onClick={() => {
            copyToClipBoard(url);
            message.open({
              type: 'success',
              content: 'Successfully copied to clipboard.'
            });
          }}
        />
      </Flex>
    )
  },
  {
    title: 'Cost',
    key: 'cost',
    width: 100,
    render: (row) =>
      `$ ${calculateCost(row.created_at, Date.now(), row.type === 'Serverless' ? row.cost / 6 : row.cost)}`
  },
  {
    title: (
      <Space>
        Estimated Savings <QuestionCircleOutlined />
      </Space>
    ),
    key: 'savings',
    width: 120,
    render: (row) =>
      row.type === 'Serverless'
        ? `${100 - Math.floor((row.cost / row.original_cost / 6) * 100) || 0}%`
        : `${100 - Math.floor((row.cost / row.original_cost) * 100) || 0}%`
  }
];

export default function Inference(props) {
  const [, setNow] = useState(Date.now());
  const [inferences, setInferences] = useState([]);
  const [selected, setSelected] = useState([]);
  const [selectedDetail, setSelectedDetail] = useState([]);
  const [filterInput, setFilterInput] = useState('');
  const [filteredInferences, setFilteredInferences] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [inferenceName, setInferenceName] = useState('');
  const [modelList, setModelList] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [fetchModelListLoading, setFetchModelListLoading] = useState(false);
  const [fetchModelLoading, setFetchModelLoading] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  const [inferenceType, setInferenceType] = useState(null);
  const [isCreateLoading, setIsCreateLoading] = useState(false);
  const [isDeployLoading, setIsDeployLoading] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);

  const fetchData = async () => {
    setFetchLoading(true);
    setSelectedDetail([]);
    setSelectedRowKeys([]);
    // TODO: User UID Value Storing in Storage (browser's)
    setNow(Date.now());
    const inferences = await getInferences(import.meta.env.VITE_TMP_USER_UID);
    inferences.sort((a, b) => b.created_at - a.created_at);
    setInferences(
      inferences.map((inference) => ({ ...inference, key: inference.uid }))
    );
    setFetchLoading(false);
  };

  const handleCancel = () => {
    setSelectedModelId(null);
    setInferenceName('');
    setSelectedModel(null);
    setIsModalOpen(false);
  };

  const fetchModelList = async () => {
    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;
    if (modelList.length) return;
    setFetchModelListLoading(true);
    const models = await getModels(user);
    if (!models)
      return messageApi.open({
        type: 'error',
        content: 'Model not found.'
      });
    setModelList(
      models.map((model) => ({
        label: `${model.name} (${model.uid})`,
        value: model.uid,
        ...model
      }))
    );
    setFetchModelListLoading(false);
  };

  const fetchModel = async (uid) => {
    setSelectedModelId(uid);
    setFetchModelLoading(true);
    const model = await getModel(uid);
    if (!model) {
      setFetchModelLoading(false);
      return messageApi.open({
        type: 'error',
        content: 'Failed to load model. Please try again.'
      });
    }
    setSelectedModel(model);
    setFetchModelLoading(false);
  };

  const handleCreateInference = async () => {
    if (
      !inferenceName ||
      !selectedModel ||
      !inferenceType ||
      !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName)
    )
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });

    setIsCreateLoading(true);

    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;
    const args = {
      user,
      name: inferenceName,
      model: selectedModel.uid,
      model_type: selectedModel.type,
      type: inferenceType,
      model_detail: {
        s3_url: selectedModel.s3_url,
        max_used_ram:
          selectedModel.max_used_ram ||
          (inferenceType === 'Spot' ? 20480 : 5120),
        ...(inferenceType === 'Spot' && {
          deployment_type:
            !selectedModel?.deploy_platform ||
            selectedModel.deploy_platform === 'Serverless'
              ? 'nodepool-2'
              : selectedModel.deploy_platform
        })
      }
    };
    const endpoint =
      inferenceType === 'Serverless'
        ? await createServerlessInference(args)
        : selectedModel.type === 'user'
          ? await createSpotInference(args)
          : await createFMInference(selectedModel.type, args);

    setIsCreateLoading(false);
    if (!endpoint)
      return messageApi.open({
        type: 'error',
        content:
          'An error occurred while creating an endpoint. Please try again.'
      });
    setIsModalOpen(false);
    await fetchData();
    return messageApi.open({
      type: 'success',
      content: 'The endpoint creation request was completed successfully.'
    });
  };

  const handleDeleteInference = async () => {
    const target = selectedDetail[0];
    if (!target) return;
    target?.streamlit_url && (await handleStreamlit('delete'));
    target.type === 'Serverless'
      ? await deleteServerlessInference({
          uid: target.uid,
          user: target.user,
          model: target.model,
          name: target.name
        })
      : target.model_type === 'user'
        ? await deleteSpotInference({
            uid: target.uid,
            user: target.user,
            name: target.name
          })
        : await deleteFMInference(target.model_type, {
            uid: target.uid,
            user: target.user,
            name: target.name
          });
    await fetchData();
    messageApi.open({
      type: 'success',
      content: 'The endpoint has been removed successfully.'
    });
  };

  const handleStreamlit = async (action) => {
    if (!selectedDetail.length) return;
    setIsDeployLoading(true);
    const isCompleted = await manageStreamlit({
      user: selectedDetail[0].user,
      uid: selectedDetail[0].uid,
      name: selectedDetail[0].name,
      model_type: selectedDetail[0].model_type,
      endpoint_url: selectedDetail[0].endpoint,
      action
    });
    setIsDeployLoading(false);
    if (!isCompleted)
      return messageApi.open({
        type: 'error',
        content:
          action === 'create'
            ? 'Streamlit was not deployed. Please try again later.'
            : 'Streamlit was not un-deployed. Please try again later.'
      });
    messageApi.open({
      type: 'success',
      content:
        action === 'create'
          ? 'Streamlit has been deployed successfully.'
          : 'Streamlit has been un-deployed successfully.'
    });
    await fetchData();
  };

  const handleUpdateModalOpen = () => {
    setInferenceName(selectedDetail[0].name || '');
    setIsUpdateModalOpen(true);
  };

  const handleUpdateInference = async () => {
    if (!inferenceName || !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName))
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });
    setIsCreateLoading(true);
    await updateInference(selected[0], { name: inferenceName });
    await fetchData();
    setIsCreateLoading(false);
    setIsUpdateModalOpen(false);
    messageApi.open({
      type: 'success',
      content: 'The endpoint has been updated successfully.'
    });
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    setFilteredInferences(
      inferences.filter((inference) =>
        inference.name.toLowerCase().includes(filterInput)
      )
    );
  }, [filterInput]);

  return (
    <>
      {contextHolder}
      <PageLayout>
        <Section>
          <Flex
            style={{ width: '100%', marginBottom: '20px' }}
            justify={'space-between'}
            align={'center'}
          >
            <div className={'section-title'}>Inference</div>
            <TableToolbox>
              <Button onClick={fetchData}>
                <SyncOutlined />
              </Button>
              <Input
                addonBefore={<SearchOutlined />}
                onChange={(e) => {
                  setFilterInput(e.target.value);
                }}
              />
              <Dropdown
                menu={{
                  items: [
                    {
                      label: 'Edit',
                      key: 'update',
                      disabled: selected.length >= 2,
                      onClick: () => handleUpdateModalOpen()
                    },
                    {
                      label: 'Deploy Streamlit',
                      key: 'deploy',
                      onClick: () => handleStreamlit('create'),
                      disabled:
                        (selectedDetail[0]?.streamlit_url &&
                          selectedDetail[0]?.streamlit_url !== '-') ||
                        selectedDetail[0]?.model_type === 'user'
                    },
                    {
                      label: 'Un-deploy Streamlit',
                      key: 'undeploy',
                      onClick: () => handleStreamlit('delete'),
                      disabled:
                        !selectedDetail[0]?.streamlit_url ||
                        selectedDetail[0].streamlit_url === '-'
                    },
                    {
                      label: (
                        <Popconfirm
                          title={'Delete it?'}
                          onConfirm={handleDeleteInference}
                        >
                          <div style={{ width: '100%' }}>Delete</div>
                        </Popconfirm>
                      ),
                      key: 'delete',
                      danger: true
                    }
                  ]
                }}
                disabled={!selectedRowKeys.length || isDeployLoading}
              >
                <Button>Actions</Button>
              </Dropdown>
              <Button
                type={'default'}
                onClick={() =>
                  window.open(selectedDetail[0].streamlit_url, '_blank')
                }
                loading={isDeployLoading}
                disabled={
                  !selectedDetail.length ||
                  !selectedDetail[0]?.streamlit_url ||
                  selectedDetail[0]?.streamlit_url === '-'
                }
              >
                <LinkOutlined />
                Streamlit
              </Button>
              <Button type={'primary'} onClick={() => setIsModalOpen(true)}>
                <PlusOutlined />
                Add New
              </Button>
            </TableToolbox>
          </Flex>
          <Table
            loading={fetchLoading}
            columns={INFERENCE_TABLE_COLUMNS}
            dataSource={filterInput ? filteredInferences : inferences}
            rowSelection={{
              selectedRowKeys,
              type: 'radio',
              onChange: (selectedRowKeys, selectedRows) => {
                setSelectedRowKeys(selectedRowKeys);
                setSelected(selectedRowKeys);
                setSelectedDetail(selectedRows);
              }
            }}
          />
        </Section>
      </PageLayout>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>New endpoint</Title>}
        open={isModalOpen}
        onCancel={handleCancel}
        footer={[
          <Button onClick={handleCancel}>Cancel</Button>,
          <Button
            type={'primary'}
            onClick={handleCreateInference}
            loading={isCreateLoading}
          >
            Create
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Enter your endpoint name</InputTitle>
          <Input
            placeholder={'Name'}
            value={inferenceName}
            onChange={(e) => setInferenceName(e.target.value)}
            status={
              inferenceName &&
              !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName) &&
              'error'
            }
          />
          {inferenceName && !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName) && (
            <>
              <ErrorMessage>
                The endpoint name can be up to 20 characters long.
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
              fetchModelListLoading ? <Spin size={'small'} /> : undefined
            }
            optionFilterProp={'label'}
            placeholder={'Model'}
            onFocus={fetchModelList}
            options={modelList}
            value={selectedModelId}
            onChange={(value) => fetchModel(value)}
          />
          {fetchModelLoading && (
            <Spin size={'default'} style={{ marginTop: '40px' }} />
          )}
          {!fetchModelLoading && selectedModel && (
            <>
              <InputTitle>Select inference type</InputTitle>
              <IconRadio
                selected={inferenceType}
                setSelected={setInferenceType}
                recommend={
                  selectedModel.deploy_platform &&
                  (selectedModel.deploy_platform === 'Serverless'
                    ? 'Serverless'
                    : 'Spot')
                }
                options={
                  selectedModel?.type === 'user'
                    ? [
                        {
                          key: 'Serverless',
                          label: 'serverless',
                          icon: CloudOutlined
                        },
                        {
                          key: 'Spot',
                          label: 'spot',
                          icon: CloudServerOutlined
                        }
                      ]
                    : [
                        {
                          key: 'Spot',
                          label: 'spot',
                          icon: CloudServerOutlined
                        }
                      ]
                }
              />
            </>
          )}
        </Flex>
      </Modal>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>Update endpoint</Title>}
        open={isUpdateModalOpen}
        confirmLoading={isCreateLoading}
        onCancel={() => setIsUpdateModalOpen(false)}
        footer={[
          <Button onClick={() => setIsUpdateModalOpen(false)}>Cancel</Button>,
          <Button
            type={'primary'}
            onClick={handleUpdateInference}
            loading={isCreateLoading}
          >
            Save
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Enter your model name</InputTitle>
          <Input
            placeholder={'Name'}
            value={inferenceName}
            onChange={(e) => setInferenceName(e.target.value)}
            status={
              inferenceName &&
              !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName) &&
              'error'
            }
          />
          {inferenceName && !/^[a-zA-Z0-9-_]{1,20}$/.test(inferenceName) && (
            <>
              <ErrorMessage>
                The endpoint name can be up to 20 characters long.
              </ErrorMessage>
              <ErrorMessage>
                Only English and special characters (-, _) can be entered.
              </ErrorMessage>
            </>
          )}
        </Flex>
      </Modal>
    </>
  );
}
