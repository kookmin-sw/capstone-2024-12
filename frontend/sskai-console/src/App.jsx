import Header from './components/Header/index.jsx';
import SideBar from './components/SideBar/index.jsx';
import { ConfigProvider, Layout } from 'antd';
import styled from 'styled-components';
import { Outlet, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard/index.jsx';
import Index from './pages/index.jsx';
import Train from './pages/Train/index.jsx';
import Inference from './pages/Inference/index.jsx';
import NewModel from './pages/NewModel/index.jsx';

const settings = {
  components: {
    Layout: {
      siderBg: '#fff',
      triggerBg: '#fff',
      triggerColor: '#000',
      headerPadding: '0px 24px'
    }
  }
};

const Container = styled.div`
  width: 100%;
  height: 100vh;
`;

function App() {
  return (
    <Container>
      <ConfigProvider theme={settings}>
        <Layout>
          <Header />
          <Layout>
            <SideBar />
            <Layout.Content style={{ height: 'calc(100vh - 64px)' }}>
              {/*<Outlet />*/}
              <Routes>
                <Route index element={<Index />} />
                <Route path={'/dashboard'} element={<Dashboard />} />
                <Route path={'/train'} element={<Train />} />
                <Route path={'/inference'} element={<Inference />} />
                <Route path={'/new-model'} element={<NewModel />} />
              </Routes>
            </Layout.Content>
          </Layout>
        </Layout>
      </ConfigProvider>
    </Container>
  );
}

export default App;
