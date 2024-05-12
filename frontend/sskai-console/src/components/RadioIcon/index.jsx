import styled from 'styled-components';
import React, { useState } from 'react';
import { Flex } from 'antd';

const Radio = styled.div`
  border: 2px solid ${(props) => (props.isSelected ? '#4096ff' : '#d9d9d9')};
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  min-height: 80px;
  cursor: pointer;
  justify-content: center;
  align-items: center;

  &:hover {
    border-color: #4096ff;
  }

  > span {
    font-weight: 600;
    color: ${(props) => (props.isSelected ? '#4096ff' : '#d9d9d9')};
  }
`;

const RecoomendLabel = styled.div`
  display: flex;
  background: #ff0000;
  margin-top: 8px;
  align-self: center;
  justify-content: center;
  align-items: center;
  padding: 2px 12px;
  border-radius: 8px;
  font-weight: 700;
  color: white;
`;

export default function RadioIcon(props) {
  const [selected, setSelected] = useState(props.recommend);
  return (
    <Flex style={{ width: '100%', ...props?.style }} gap={10}>
      {props.options.map((option) => (
        <Flex vertical style={{ flex: 1 }}>
          <Radio
            key={option.key}
            id={option.key}
            isSelected={selected === option.key}
            onClick={(e) => setSelected(option.key)}
          >
            {
              <option.icon
                style={{
                  fontSize: '40px',
                  color: selected === option.key ? '#4096ff' : '#d9d9d9'
                }}
              />
            }
            <span>{option.label.toUpperCase()}</span>
          </Radio>
          {props.recommend === option.key && (
            <RecoomendLabel>Recommend</RecoomendLabel>
          )}
        </Flex>
      ))}
    </Flex>
  );
}
