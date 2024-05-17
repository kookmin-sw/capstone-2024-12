import styled from 'styled-components';

export const PageLayout = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 100%;
  height: 100%;
  padding: 20px 24px;
  overflow: auto;
`;

export const TableToolbox = styled.div`
  display: flex;
  gap: 10px;
`;

export const Title = styled.div`
  font-size: 18px;
  font-weight: 500;
`;

export const InputTitle = styled(Title)`
  margin-top: 20px;
  margin-bottom: 8px;
`;

export const ErrorMessage = styled.div`
  color: #ff4d4f;
`;
