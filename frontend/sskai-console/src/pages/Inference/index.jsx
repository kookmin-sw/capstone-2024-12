import { PageLayout } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import { Table } from 'antd';

const INFERENCE_TABLE_COLUMNS = [
  {
    title: 'Name',
    dataIndex: 'name',
    key: 'name'
  },
  {
    title: 'Type',
    dataIndex: 'type',
    key: 'type'
  },
  {
    title: 'Endpoint URL',
    dataIndex: 'endpoint',
    key: 'endpoint'
  },
  {
    title: 'Cost',
    dataIndex: 'cost',
    key: 'cost'
  },
  {
    title: 'Estimated Savings',
    dataIndex: 'savings',
    key: 'savings'
  }
];

export default function Inference(props) {
  return (
    <PageLayout>
      <Section>
        <Table columns={INFERENCE_TABLE_COLUMNS} />
      </Section>
    </PageLayout>
  );
}
