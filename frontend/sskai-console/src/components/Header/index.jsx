import { QuestionCircleOutlined, SettingOutlined } from '@ant-design/icons';
import { Button, Layout } from 'antd';
import styled from 'styled-components';

const Logo = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  color: #fff;
  font-size: 24px;
  font-weight: 900;
  gap: 8px;
`;

const Menu = styled.div`
  display: flex;
  align-items: center;
  margin-left: auto;
  gap: 10px;
`;

const ButtonGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 15px;
  .outline-white {
    > svg {
      fill: #fff;
    }
  }
`;

const User = styled.div`
  margin-left: 10px;
  color: white;
  font-size: 16px;
  > .user-name {
    font-weight: bold;
  }
`;

export default function Header(props) {
  return (
    <Layout.Header style={{ display: 'flex' }}>
      <Logo>
        <img src={'https://placehold.co/50x50'} alt={''} width={'50px'} />
        SSKAI
      </Logo>
      <Menu>
        <ButtonGroup>
          <Button
            size={'small'}
            type={'text'}
            icon={<QuestionCircleOutlined className={'outline-white'} />}
          />
          <Button
            size={'small'}
            type={'text'}
            icon={<SettingOutlined className={'outline-white'} />}
          />
        </ButtonGroup>
        <User>
          Hello, <span className={'user-name'}>John Doe</span>
        </User>
      </Menu>
    </Layout.Header>
  );
}
