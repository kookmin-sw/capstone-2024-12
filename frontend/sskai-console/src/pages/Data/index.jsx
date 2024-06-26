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
  Upload
} from 'antd';
import { useEffect, useState } from 'react';
import {
  createData,
  deleteData,
  getData,
  updateData,
  uploadS3,
  uploadS3Multipart
} from '../../api/index.jsx';
import { SearchOutlined, PlusOutlined, InboxOutlined } from '@ant-design/icons';
import { formatTimestamp } from '../../utils/index.jsx';

const DATA_TABLE_COLUMNS = [
  {
    title: 'Name',
    dataIndex: 'name',
    width: 600
  },
  {
    title: 'Creation Time',
    dataIndex: 'created_at',
    width: 300,
    render: (timestamp) => formatTimestamp(timestamp)
  }
];

export default function Data() {
  const [data, setData] = useState([]);
  const [selected, setSelected] = useState('');
  const [filterInput, setFilterInput] = useState('');
  const [filteredData, setFilteredData] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dataName, setDataName] = useState('');
  const [dataFile, setDataFile] = useState([]);
  const [createdData, setCreatedData] = useState(null);
  const [messageApi, contextHolder] = message.useMessage();
  const [fetchLoading, setFetchLoading] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedRows, setSelectedRows] = useState([]);

  const uploadSettings = {
    maxCount: 1,
    beforeUpload: (file) => {
      const isModel = ['application/zip'].includes(file.type);
      if (!isModel) {
        messageApi.open({
          type: 'error',
          content: 'Data file can only be uploaded as .zip.'
        });
        return Upload.LIST_IGNORE;
      }
      setDataFile([file]);
      return false;
    },
    fileList: dataFile,
    onRemove: () => {
      setDataFile([]);
    }
  };

  const fetchData = async () => {
    setFetchLoading(true);
    // TODO: User UID Value Storing in Storage (browser's)
    const dataList = await getData(import.meta.env.VITE_TMP_USER_UID);
    dataList.sort((a, b) => b.created_at - a.created_at);
    setData(dataList.map((data) => ({ ...data, key: data.uid })));
    setFetchLoading(false);
  };

  const handleCreateData = async () => {
    // TODO: User UID Value Storing in Storage (browser's)
    const user = import.meta.env.VITE_TMP_USER_UID;

    if (!dataName || !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName))
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });

    if (!dataFile.length)
      return messageApi.open({
        type: 'error',
        content: 'Please upload your model file.'
      });

    setLoading(true);

    const data =
      createdData ||
      (await createData({
        user,
        name: dataName
      }));

    if (!data) {
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while creating the your data. Please try again in a few minutes.'
      });
    }

    if (!createData) setCreatedData(data);

    const uploaded =
      dataFile[0].size < 5 * 1024 * 1024 * 1024 // 5GB
        ? await uploadS3('data', user, data.uid, dataFile[0])
        : await uploadS3Multipart('data', user, data.uid, dataFile[0]);

    if (!uploaded) {
      await deleteData(data.uid);
      setCreatedData(null);
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while uploading the your data. Please try again in a few minutes.'
      });
    }

    const isUpdated = await updateData(data.uid, {
      s3_url: uploaded
    });

    if (!isUpdated) {
      await deleteData(data.uid);
      setCreatedData(null);
      setLoading(false);
      return messageApi.open({
        type: 'error',
        content:
          'Sorry, an error occurred while updating the your data. Please try again in a few minutes.'
      });
    }

    await fetchData();
    setCreatedData(null);
    setLoading(false);
    handleCancel();

    return messageApi.open({
      type: 'success',
      content: 'The data has been created successfully.'
    });
  };

  const handleCancel = () => {
    setDataName('');
    setDataFile([]);
    setIsModalOpen(false);
    setIsUpdateModalOpen(false);
  };

  const handleDeleteData = async () => {
    await Promise.all(selected.map((uid) => deleteData(uid)));
    await fetchData();
    messageApi.open({
      type: 'success',
      content: 'The data has been removed successfully.'
    });
  };

  const handleUpdateModalOpen = (uid) => {
    setDataName(data.filter((item) => item.uid === uid).pop().name || '');
    setIsUpdateModalOpen(true);
  };

  const handleUpdateData = async () => {
    if (!dataName || !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName))
      return messageApi.open({
        type: 'error',
        content:
          'Please check that you have entered all items according to the entry conditions.'
      });
    setLoading(true);
    await updateData(selected[0], { name: dataName });
    await fetchData();
    setLoading(false);
    handleCancel();
    return messageApi.open({
      type: 'success',
      content: 'The data has been updated successfully.'
    });
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    setFilteredData(
      data.filter((data) => data.name.toLowerCase().includes(filterInput))
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
            <div className={'section-title'}>Data</div>
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
                      label: 'Download',
                      key: 'download',
                      disabled: selected.length >= 2,
                      onClick: () =>
                        window.open(selectedRows[0]?.s3_url, '_blank')
                    },
                    {
                      label: 'Edit',
                      key: 'update',
                      disabled: selected.length >= 2,
                      onClick: () => handleUpdateModalOpen(selected[0])
                    },
                    {
                      label: (
                        <Popconfirm
                          title={'Delete it?'}
                          onConfirm={handleDeleteData}
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
            columns={DATA_TABLE_COLUMNS}
            dataSource={filterInput ? filteredData : data}
            rowSelection={{
              type: 'checkbox',
              onChange: (selectedRowKeys, selectedRows) => {
                setSelected(selectedRowKeys);
                setSelectedRows(selectedRows);
              }
            }}
          />
        </Section>
      </PageLayout>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>Add new data</Title>}
        open={isModalOpen}
        confirmLoading={loading}
        onCancel={handleCancel}
        footer={[
          <Button onClick={handleCancel}>Cancel</Button>,
          <Button type={'primary'} onClick={handleCreateData} loading={loading}>
            Upload
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Enter your data name</InputTitle>
          <Input
            placeholder={'Name'}
            value={dataName}
            onChange={(e) => setDataName(e.target.value)}
            status={
              dataName && !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName) && 'error'
            }
          />
          {dataName && !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName) && (
            <>
              <ErrorMessage>
                The data name can be up to 20 characters long.
              </ErrorMessage>
              <ErrorMessage>
                Only English and special characters (-, _) can be entered.
              </ErrorMessage>
            </>
          )}
          <InputTitle>Upload your data</InputTitle>
          <Upload.Dragger {...uploadSettings} disabled={loading}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag file to this area to upload
            </p>
          </Upload.Dragger>
        </Flex>
      </Modal>
      <Modal
        title={<Title style={{ fontWeight: 600 }}>Update data</Title>}
        open={isUpdateModalOpen}
        confirmLoading={loading}
        onCancel={handleCancel}
        footer={[
          <Button onClick={handleCancel}>Cancel</Button>,
          <Button type={'primary'} onClick={handleUpdateData} loading={loading}>
            Save
          </Button>
        ]}
      >
        <Flex vertical style={{ marginBottom: '20px' }}>
          <InputTitle>Enter your data name</InputTitle>
          <Input
            placeholder={'Name'}
            value={dataName}
            onChange={(e) => setDataName(e.target.value)}
            status={
              dataName && !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName) && 'error'
            }
          />
          {dataName && !/^[a-zA-Z0-9-_]{1,20}$/.test(dataName) && (
            <>
              <ErrorMessage>
                The data name can be up to 20 characters long.
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
