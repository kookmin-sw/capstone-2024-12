import { QuestionCircleOutlined, SettingOutlined } from '@ant-design/icons';
import { Button, Layout, message } from 'antd';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';
import LogoSVG from '/logo.svg';

const Logo = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  color: #fff;
  font-size: 24px;
  font-weight: 900;
  gap: 8px;
  letter-spacing: -1px;
  cursor: pointer;

  > img {
    -webkit-user-drag: none;
    user-drag: none;
  }
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
  const navigate = useNavigate();
  return (
    <Layout.Header style={{ display: 'flex' }}>
      <Logo
        onClick={() => {
          navigate('/dashboard');
        }}
      >
        <img src={LogoSVG} alt={''} width={'50px'} />
        SSKAI
      </Logo>
      <Menu>
        <ButtonGroup>
          <Button
            size={'small'}
            type={'text'}
            icon={<QuestionCircleOutlined className={'outline-white'} />}
            onClick={() =>
              window.open(
                'https://github.com/kookmin-sw/capstone-2024-12',
                '_blank'
              )
            }
          />
          <Button
            size={'small'}
            type={'text'}
            icon={<SettingOutlined className={'outline-white'} />}
            onClick={() => {
              message.open({
                type: 'error',
                content: "You're not authorized. Please check your permission."
              });
            }}
          />
        </ButtonGroup>
        <User>
          Hello, <span className={'user-name'}>John Doe</span>
        </User>
      </Menu>
    </Layout.Header>
  );
}
