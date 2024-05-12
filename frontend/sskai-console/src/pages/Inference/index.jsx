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
  Select,
  Space,
  Spin,
  Table,
  Tag
} from 'antd';
import { useEffect, useState } from 'react';
import { getInferences, getModels } from '../../api/index.jsx';
import {
  PlusOutlined,
  SearchOutlined,
  SyncOutlined,
  CopyOutlined,
  QuestionCircleOutlined,
  CloudOutlined,
  CloudServerOutlined
} from '@ant-design/icons';
import axios from 'axios';
import RadioIcon from '../../components/RadioIcon/index.jsx';
import { copyToClipBoard } from '../../utils/index.jsx';

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
    dataIndex: 'cost',
    key: 'cost',
    width: 100,
    render: (item) => `$ ${item}`
  },
  {
    title: (
      <Space>
        Estimated Savings <QuestionCircleOutlined />
      </Space>
    ),
    dataIndex: 'savings',
    key: 'savings',
    width: 120,
    render: () => '0%'
  }
];

export default function Inference(props) {
  const [, setNow] = useState(Date.now());
  const [inferences, setInferences] = useState([]);
  const [selected, setSelected] = useState('');
  const [filterInput, setFilterInput] = useState('');
  const [filteredInferences, setFilteredInferences] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [inferenceName, setInferenceName] = useState('');
  const [modelList, setModelList] = useState([]);
  const [fetchModelLoading, setFetchModelLoading] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  // TODO: DELETE START (Only For Testing)
  const [isTestOpen, setIsTestOpen] = useState(false);
  const [testForm, setTestForm] = useState({});
  const [isTestLoading, setIsTestLoading] = useState(false);
  // TODO: DELETE END

  const fetchData = async () => {
    setFetchLoading(true);
    // TODO: User UID Value Storing in Storage (browser's)
    setNow(Date.now());
    const models = await getInferences(import.meta.env.VITE_TMP_USER_UID);
    setInferences(models.map((model) => ({ ...model, key: model.uid })));
    setFetchLoading(false);
  };

  // TODO: DELETE START (Only For Testing)
  const handleCancel = () => {
    const isReset = confirm('Reset?');
    if (!isReset) return;
    setTestForm({});
  };
  const handleTest = async () => {
    if (!testForm.endpoint)
      return message.open({
        type: 'error',
        content: 'Please enter Endpoint URL.'
      });
    try {
      JSON.parse(testForm.body);
    } catch (err) {
      return message.open({
        type: 'error',
        content: 'Please enter in JSON format.'
      });
    }
    setIsTestLoading(true);
    const res = await axios
      .post(testForm.endpoint, JSON.parse(testForm.body))
      .catch((err) => err);
    if (!res?.data) {
      console.log(res);
      setIsTestLoading(false);
      return message.open({
        type: 'error',
        content: 'Occurred error. Please check your browser console.'
      });
    }
    setTestForm({ ...testForm, res: JSON.stringify(res.data, null, 4) });
    setIsTestLoading(false);
  };
  // TODO: DELETE END

  const fetchModelList = async () => {
    if (modelList.length) return;
    // TODO: User UID Value Storing in Storage (browser's)
    setFetchModelLoading(true);
    const models = await getModels(import.meta.env.VITE_TMP_USER_UID);
    if (!models)
      return messageApi.open({
        type: 'error',
        content: 'Model not found.'
      });
    setModelList(
      models.map((model) => ({
        label: `${model.name} (${model.uid})`,
        value: model.uid
      }))
    );
    setFetchModelLoading(false);
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
                      disabled: selected.length >= 2
                    },
                    {
                      label: 'Delete',
                      key: 'delete',
                      danger: true
                    }
                  ]
                }}
                disabled={!selected.length}
              >
                <Button>Actions</Button>
              </Dropdown>
              <Button type={'primary'} onClick={() => setIsTestOpen(true)}>
                <PlusOutlined />
                Test
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
              type: 'checkbox',
              onChange: (selectedRowKeys) => {
                setSelected(selectedRowKeys);
              }
            }}
          />
        </Section>
      </PageLayout>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>Test Form</Title>}
        open={isTestOpen}
        onCancel={() => setIsTestOpen(false)}
        footer={[
          <Button onClick={handleCancel}>Reset</Button>,
          <Button type={'primary'} onClick={handleTest} loading={isTestLoading}>
            Send
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Input Endpoint URL</InputTitle>
          <Input
            placeholder={'https://example.com'}
            value={testForm.endpoint}
            onChange={(e) =>
              setTestForm({ ...testForm, endpoint: e.target.value })
            }
            disabled={isTestLoading}
          />
          <InputTitle>Input Body (JSON)</InputTitle>
          <Input.TextArea
            rows={7}
            placeholder={'{"hello": "world"}'}
            style={{ resize: 'none' }}
            value={testForm.body}
            onChange={(e) => setTestForm({ ...testForm, body: e.target.value })}
            disabled={isTestLoading}
          />
          <InputTitle>Output Body (JSON)</InputTitle>
          <Input.TextArea
            readOnly
            value={testForm.res}
            rows={7}
            style={{
              resize: 'none',
              whiteSpace: 'pre-wrap',
              background: '#f4f4f4',
              border: '1px solid #ccc',
              padding: '10px',
              borderRadius: '5px',
              color: '#333',
              overflowX: 'auto'
            }}
          />
        </Flex>
      </Modal>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>New inference</Title>}
        open={isModalOpen}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Input your endpoint name</InputTitle>
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
            options={modelList}
          />
          <InputTitle>Select inference type</InputTitle>
          <RadioIcon
            recommend={'spot'}
            options={[
              {
                key: 'serverless',
                label: 'serverless',
                icon: CloudOutlined
              },
              {
                key: 'spot',
                label: 'spot',
                icon: CloudServerOutlined
              }
            ]}
          />
        </Flex>
      </Modal>
    </>
  );
}
