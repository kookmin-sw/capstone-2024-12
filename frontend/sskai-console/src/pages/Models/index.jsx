import { PageLayout, TableToolbox } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import { Button, Dropdown, Flex, Input, Table, Tag } from 'antd';
import { useEffect, useState } from 'react';
import { getModels } from '../../api/index.jsx';
import { SearchOutlined, PlusOutlined } from '@ant-design/icons';
import { formatTimestamp } from '../../utils/index.jsx';
import { useNavigate } from 'react-router-dom';

const MODEL_TABLE_COLUMNS = [
  {
    title: 'Name',
    dataIndex: 'name',
    width: 300
  },
  {
    title: 'Type',
    dataIndex: 'type',
    width: 300,
    render: (type) => {
      const color = type === 'user' ? 'green' : 'geekblue';
      return (
        <Tag color={color} key={type}>
          {type.toUpperCase()}
        </Tag>
      );
    }
  },
  {
    title: 'Creation Time',
    dataIndex: 'created_at',
    width: 300,
    render: (timestamp) => formatTimestamp(timestamp)
  }
];

export default function Models(props) {
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState('');
  const [filterInput, setFilterInput] = useState('');
  const [filteredModel, setFilteredModel] = useState([]);

  const fetchData = async () => {
    // TODO: User UID Value Storing in Storage (browser's)
    const models = await getModels(import.meta.env.VITE_TMP_USER_UID);
    setModels(models.map((model) => ({ ...model, key: model.uid })));
    console.log(models);
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    setFilteredModel(
      models.filter((model) => model.name.toLowerCase().includes(filterInput))
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
          <div className={'section-title'}>Models</div>
          <TableToolbox>
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
            <Button type={'primary'} onClick={() => navigate('/new-model')}>
              <PlusOutlined />
              Add New
            </Button>
          </TableToolbox>
        </Flex>
        <Table
          columns={MODEL_TABLE_COLUMNS}
          dataSource={filterInput ? filteredModel : models}
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
