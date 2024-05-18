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
  Table,
  Tag
} from 'antd';
import { useEffect, useState } from 'react';
import { deleteModel, getModels, updateModel } from '../../api/index.jsx';
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
          {(type === 'user' ? 'user' : 'fm').toUpperCase()}
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

export default function Model(props) {
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState([]);
  const [filterInput, setFilterInput] = useState('');
  const [filteredModel, setFilteredModel] = useState([]);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [modelName, setModelName] = useState('');
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();

  const fetchData = async () => {
    setFetchLoading(true);
    // TODO: User UID Value Storing in Storage (browser's)
    const models = await getModels(import.meta.env.VITE_TMP_USER_UID);
    models.sort((a, b) => b.created_at - a.created_at);
    setModels(models.map((model) => ({ ...model, key: model.uid })));
    setFetchLoading(false);
  };

  const handleDeleteModel = async () => {
    await Promise.all(selected.map((uid) => deleteModel(uid)));
    await fetchData();
    messageApi.open({
      type: 'success',
      content: 'The model has been removed successfully.'
    });
  };

  const handleCancel = () => {
    setModelName('');
    setIsUpdateModalOpen(false);
  };

  const handleUpdateModalOpen = (uid) => {
    setModelName(models.filter((item) => item.uid === uid).pop().name || '');
    setIsUpdateModalOpen(true);
  };

  const handleUpdateModel = async () => {
    if (!modelName || !/^[a-zA-Z0-9-_]{1,20}$/.test(modelName))
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });
    setLoading(true);
    await updateModel(selected[0], { name: modelName });
    await fetchData();
    setLoading(false);
    handleCancel();
    messageApi.open({
      type: 'success',
      content: 'The model has been updated successfully.'
    });
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
    <>
      {contextHolder}
      <PageLayout>
        <Section>
          <Flex
            style={{ width: '100%', marginBottom: '20px' }}
            justify={'space-between'}
            align={'center'}
          >
            <div className={'section-title'}>Model</div>
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
                      onClick: () => handleUpdateModalOpen(selected[0])
                    },
                    {
                      label: (
                        <Popconfirm
                          title={'Delete it?'}
                          onConfirm={handleDeleteModel}
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
              <Button type={'primary'} onClick={() => navigate('/new-model')}>
                <PlusOutlined />
                Add New
              </Button>
            </TableToolbox>
          </Flex>
          <Table
            loading={fetchLoading}
            columns={MODEL_TABLE_COLUMNS}
            dataSource={filterInput ? filteredModel : models}
            rowSelection={{
              type: 'radio',
              onChange: (selectedRowKeys) => {
                setSelected(selectedRowKeys);
              }
            }}
          />
        </Section>
      </PageLayout>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>Update model</Title>}
        open={isUpdateModalOpen}
        confirmLoading={loading}
        onCancel={handleCancel}
        footer={[
          <Button onClick={handleCancel}>Cancel</Button>,
          <Button
            type={'primary'}
            onClick={handleUpdateModel}
            loading={loading}
          >
            Save
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Enter your model name</InputTitle>
          <Input
            placeholder={'Name'}
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            status={
              modelName && !/^[a-zA-Z0-9-_]{1,20}$/.test(modelName) && 'error'
            }
          />
          {modelName && !/^[a-zA-Z0-9-_]{1,20}$/.test(modelName) && (
            <>
              <ErrorMessage>
                The model name can be up to 20 characters long.
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
