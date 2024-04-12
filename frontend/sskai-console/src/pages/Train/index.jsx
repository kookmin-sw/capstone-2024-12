import { PageLayout } from '../styles.jsx';
import { Section } from '../../components/Section/index.jsx';
import { Table } from 'antd';

const TRAIN_TABLE_COLUMNS = [
  {
    title: 'Name',
    dataIndex: 'name',
    key: 'name'
  },
  {
    title: 'Status',
    dataIndex: 'status',
    key: 'status'
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
  },
  {
    title: 'Elapsed Time',
    dataIndex: 'time',
    key: 'time'
  }
];

export default function Train(props) {
  return (
    <PageLayout>
      <Section>
        <Table columns={TRAIN_TABLE_COLUMNS} />
      </Section>
    </PageLayout>
  );
}
