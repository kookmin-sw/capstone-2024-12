import {
  ErrorMessage,
  InputTitle,
  PageLayout,
  TableToolbox,
  Title
} from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import {
  Badge,
  Button,
  Dropdown,
  Flex,
  Input,
  InputNumber,
  message,
  Modal,
  Select,
  Space,
  Spin,
  Table
} from 'antd';
import { useNavigate } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';
import { getData, getModels, getTrains } from '../../api/index.jsx';
import {
  PlusOutlined,
  SearchOutlined,
  SyncOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';
import { calculateDuration } from '../../utils/index.jsx';

const STATUS_BADGE_MAPPER = {
  Processing: 'processing',
  Pause: 'warning',
  Stopped: 'error'
};

const TRAIN_TABLE_COLUMNS = (now) => [
  {
    title: 'Name',
    dataIndex: 'name',
    key: 'name',
    width: 300
  },
  {
    title: 'Status',
    dataIndex: 'status',
    key: 'status',
    render: (item) => (
      <Badge status={STATUS_BADGE_MAPPER[item] || 'default'} text={item} />
    ),
    width: 150
  },
  {
    title: 'Cost',
    dataIndex: 'cost',
    key: 'cost',
    render: (item) => `$ ${item}`,
    width: 150
  },
  {
    title: (
      <Space>
        Estimated Savings <QuestionCircleOutlined />
      </Space>
    ),
    dataIndex: 'savings',
    key: 'savings',
    width: 200,
    render: () => '0%'
  },
  {
    title: 'Elapsed Time',
    dataIndex: 'created_at',
    key: 'time',
    width: 300,
    render: (time) => calculateDuration(time, now)
  }
];

export default function Train(props) {
  const navigate = useNavigate();

  const [now, setNow] = useState(Date.now());
  const [trains, setTrains] = useState([]);
  const [selected, setSelected] = useState('');
  const [filterInput, setFilterInput] = useState('');
  const [filteredTrains, setFilteredTrains] = useState([]);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [trainName, setTrainName] = useState('');
  const [modelList, setModelList] = useState([]);
  const [fetchModelLoading, setFetchModelLoading] = useState(false);
  const [dataList, setDataList] = useState([]);
  const [fetchDataLoading, setFetchDataLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedData, setSelectedData] = useState(null);
  const [epochs, setEpochs] = useState(5);
  const [messageApi, contextHolder] = message.useMessage();

  const fetchData = async () => {
    setFetchLoading(true);
    setNow(Date.now());
    // TODO: User UID Value Storing in Storage (browser's)
    const models = await getTrains(import.meta.env.VITE_TMP_USER_UID);
    setTrains(models.map((model) => ({ ...model, key: model.uid })));
    setFetchLoading(false);
  };

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
    setDataList(
      data.map((d) => ({
        label: `${d.name} (${d.uid})`,
        value: d.uid
      }))
    );
    setFetchDataLoading(false);
  };

  const handleCancel = () => {
    setTrainName('');
    setSelectedModel(null);
    setSelectedData(null);
    setModelList([]);
    setDataList([]);
    setIsModalOpen(false);
  };

  const handleCreateTrain = async () => {
    if (
      !trainName ||
      !/^[a-zA-Z0-9-_]{1,20}$/.test(trainName) ||
      !selectedModel ||
      !selectedData ||
      !epochs
    )
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    setFilteredTrains(
      trains.filter((train) => train.name.toLowerCase().includes(filterInput))
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
            <div className={'section-title'}>Train</div>
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
              <Button type={'primary'} onClick={() => setIsModalOpen(true)}>
                <PlusOutlined />
                Add New
              </Button>
            </TableToolbox>
          </Flex>
          <Table
            loading={fetchLoading}
            columns={TRAIN_TABLE_COLUMNS(now)}
            dataSource={filterInput ? filteredTrains : trains}
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
        title={<Title style={{ fontWeight: 600 }}>New model training</Title>}
        open={isModalOpen}
        onCancel={handleCancel}
        footer={[
          <Button onClick={handleCancel}>Cancel</Button>,
          <Button type={'primary'} onClick={handleCreateTrain}>
            Start
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Input your train name</InputTitle>
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
            value={selectedModel}
            onChange={(value) => setSelectedModel(value)}
            options={modelList}
          />
          <InputTitle>Select your data</InputTitle>
          <Select
            showSearch
            notFoundContent={
              fetchDataLoading ? <Spin size={'small'} /> : undefined
            }
            optionFilterProp={'label'}
            placeholder={'Data'}
            onFocus={fetchDataList}
            value={selectedData}
            onChange={(value) => setSelectedData(value)}
            options={dataList}
          />
          <InputTitle>Epochs</InputTitle>
          <InputNumber
            style={{ width: '100%' }}
            value={epochs}
            min={1}
            defaultValue={5}
            onChange={(value) => setEpochs(Number(value) || 5)}
          />
        </Flex>
      </Modal>
    </>
  );
}
