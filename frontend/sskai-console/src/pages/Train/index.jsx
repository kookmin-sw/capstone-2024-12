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
  message,
  Popconfirm,
  Space,
  Table
} from 'antd';
import { useEffect, useState } from 'react';
import { deleteTrain, getTrains } from '../../api/index.jsx';
import {
  PlusOutlined,
  SearchOutlined,
  SyncOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';
import { calculateDuration } from '../../utils/index.jsx';
import TrainCreateModal from '../NewModel/TrainCreateModal.jsx';

const STATUS_BADGE_MAPPER = {
  Running: 'processing',
  Completed: 'success'
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
    key: 'time',
    width: 300,
    render: (row) =>
      calculateDuration(
        row.created_at,
        row.status === 'Completed' ? row.updated_at : now
      )
  }
];

export default function Train(props) {
  const [now, setNow] = useState(Date.now());
  const [trains, setTrains] = useState([]);
  const [selected, setSelected] = useState('');
  const [selectedDetail, setSelectedDetail] = useState([]);
  const [filterInput, setFilterInput] = useState('');
  const [filteredTrains, setFilteredTrains] = useState([]);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  const fetchData = async () => {
    setFetchLoading(true);
    setNow(Date.now());
    // TODO: User UID Value Storing in Storage (browser's)
    const trains = await getTrains(import.meta.env.VITE_TMP_USER_UID);
    trains.sort((a, b) => b.created_at - a.created_at);
    setTrains(trains.map((train) => ({ ...train, key: train.uid })));
    setFetchLoading(false);
  };

  const handleDeleteTrain = async () => {
    if (!selected.length) return;
    await deleteTrain(selected[0], selectedDetail[0]?.status);
    await fetchData();
    messageApi.open({
      type: 'success',
      content: 'The train has been removed successfully.'
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
                      label: (
                        <Popconfirm
                          title={'Delete it?'}
                          onConfirm={handleDeleteTrain}
                        >
                          <div style={{ width: '100%' }}>Delete</div>
                        </Popconfirm>
                      ),

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
              type: 'radio',
              onChange: (selectedRowKeys, selectedRows) => {
                setSelected(selectedRowKeys);
                setSelectedDetail(selectedRows);
              }
            }}
          />
        </Section>
      </PageLayout>
      <TrainCreateModal
        isModalOpen={isModalOpen}
        setIsModalOpen={setIsModalOpen}
        messageApi={messageApi}
      />
    </>
  );
}
