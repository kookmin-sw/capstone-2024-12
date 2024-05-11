import { PageLayout, TableToolbox } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import { Badge, Button, Dropdown, Flex, Input, Space, Table } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getTrains } from '../../api/index.jsx';
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

  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     setNow(Date.now());
  //   }, 1000);
  //
  //   return () => clearInterval(interval);
  // }, []);

  const fetchData = async () => {
    setFetchLoading(true);
    // TODO: User UID Value Storing in Storage (browser's)
    setNow(Date.now());
    const models = await getTrains(import.meta.env.VITE_TMP_USER_UID);
    setTrains(models.map((model) => ({ ...model, key: model.uid })));
    setFetchLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    setFilteredTrains(
      trains.filter((model) => model.name.toLowerCase().includes(filterInput))
    );
  }, [filterInput]);

  return (
    <PageLayout>
      <Section>
        <Flex
          style={{ width: '100%', marginBottom: '20px' }}
          justify={'space-between'}
          align={'center'}
        >
          <div className={'section-title'}>Trains</div>
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
            <Button type={'primary'}>
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
  );
}
