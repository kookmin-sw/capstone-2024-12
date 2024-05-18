import { PageLayout } from '../styles.jsx';
import styled from 'styled-components';
import { Flex, Progress, Space, Table } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useEffect, useState } from 'react';
import { Section } from '../../components/Section/index.jsx';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import CountUp from 'react-countup';

const Title = styled.div`
  display: flex;
  font-size: 20px;
  font-weight: 500;
`;

const SubTitle = styled.div`
  font-size: 16px;
`;

const ProgressWrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  margin-bottom: 10px;

  > span {
    display: flex;
    width: 100%;
    justify-content: end;
    font-size: 12px;
    margin-top: 4px;
    padding-right: 6px;
  }
`;

const Cost = styled.div`
  font-size: 30px;
  font-weight: 600;
`;

const LOG_TABLE_COLUMNS = [
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
    title: 'Recent Job',
    dataIndex: 'job',
    key: 'job'
  },
  {
    title: 'Last Time',
    dataIndex: 'time',
    key: 'time'
  },
  {
    title: 'Action',
    key: 'action',
    render: () => (
      <Space size="middle">
        <a>Data</a>
        <a>Train</a>
        <a>Inference</a>
      </Space>
    )
  }
];

const EXAMPLE_SAVINGS = [
  {
    month: 'Jan',
    savings: 94
  },
  {
    month: 'Feb',
    savings: 38
  },
  {
    month: 'Mar',
    savings: 68
  },
  {
    month: 'Apr',
    savings: 75
  },
  {
    month: 'May',
    savings: 89
  },
  {
    month: 'Jun',
    savings: 91
  }
];

const EXAMPLE_COST = [
  {
    month: 'Jan',
    cost: 935.42
  },
  {
    month: 'Feb',
    cost: 854.71
  },
  {
    month: 'Mar',
    cost: 905.16
  },
  {
    month: 'Apr',
    cost: 836.29
  },
  {
    month: 'May',
    cost: 921.83
  },
  {
    month: 'Jun',
    cost: 992.15
  }
];

export default function Dashboard(props) {
  const [savingsPercent, setSavingsPercent] = useState({
    time: 0,
    cost: 0
  });

  useEffect(() => {
    setSavingsPercent({
      time: 95,
      cost: 95
    });
  }, []);

  return (
    <PageLayout>
      <Flex gap={24}>
        <Section>
          <Flex>
            <Flex
              style={{ width: '100%', marginBottom: '24px' }}
              justify={'space-between'}
            >
              <Title>Savings</Title>
              <QuestionCircleOutlined
                style={{ fontSize: '16px', color: 'rgba(0, 0, 0, 0.45)' }}
              />
            </Flex>
          </Flex>
          <ProgressWrapper>
            <SubTitle>Time Savings</SubTitle>
            <Progress percent={savingsPercent.time} />
            <span>Total usage time: 99:99:99</span>
          </ProgressWrapper>
          <ProgressWrapper>
            <SubTitle>Cost Savings</SubTitle>
            <Progress percent={savingsPercent.cost} />
            <span>Total usage cost: $999.99</span>
          </ProgressWrapper>
          <ResponsiveContainer height={180}>
            <LineChart data={EXAMPLE_SAVINGS} margin={{ top: 30 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="savings" stroke="#1677ff" />
            </LineChart>
          </ResponsiveContainer>
        </Section>
        <Section>
          <Flex vertical={true}>
            <Flex
              style={{ width: '100%', marginBottom: '24px' }}
              justify={'space-between'}
            >
              <Title>Cost Summary</Title>
            </Flex>
            <Flex
              vertical={false}
              gap={10}
              align={'center'}
              style={{ marginBottom: '8px' }}
            >
              <SubTitle>Month-to-date cost</SubTitle>
              <QuestionCircleOutlined
                style={{ fontSize: '16px', color: 'rgba(0, 0, 0, 0.45)' }}
              />
            </Flex>
            <Cost>
              $ <CountUp end={999} duration={1} />.
              <CountUp end={99} duration={2} />
            </Cost>
            <ResponsiveContainer height={280}>
              <BarChart data={EXAMPLE_COST} margin={{ top: 30 }}>
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="cost" fill="#1677ff" />
              </BarChart>
            </ResponsiveContainer>
          </Flex>
        </Section>
      </Flex>
      <Section>
        <Flex vertical={false}>
          <Flex
            style={{ width: '100%', marginBottom: '24px' }}
            justify={'space-between'}
          >
            <Title>Progress</Title>
          </Flex>
        </Flex>
        <Table columns={LOG_TABLE_COLUMNS} />
      </Section>
    </PageLayout>
  );
}
