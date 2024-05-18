import styled from 'styled-components';
import React, { useEffect, useState } from 'react';
import { Flex, message, Spin } from 'antd';

const Radio = styled.div`
  border: 2px solid ${(props) => (props.isSelected ? '#4096ff' : '#d9d9d9')};
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  min-height: 85px;
  cursor: pointer;
  justify-content: center;
  align-items: center;
  gap: 4px;
  &:hover {
    border-color: #4096ff;
  }

  > span {
    font-weight: 600;
    color: ${(props) => (props.isSelected ? '#4096ff' : '#9d9d9d')};
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

const Grid = styled.div`
  position: relative;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
`;

const OverBlur = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  position: absolute;
  width: 105%;
  height: 105%;
  border-radius: 8px;
  backdrop-filter: blur(4px);
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
`;

export default function IconRadio(props) {
  const { selected, setSelected } = props;
  useEffect(() => {
    if (props.recommend) setSelected(props.recommend);
  }, [props.recommend]);

  if (props?.options.length > 3)
    return (
      <Grid>
        {props.loading && (
          <OverBlur>
            <Spin size={'large'} />
          </OverBlur>
        )}
        {props.options.map((option) => (
          <Flex vertical>
            <Radio
              key={option.key}
              id={option.key}
              isSelected={selected === option.key}
              onClick={
                option?.disabled
                  ? () =>
                      message.open({
                        type: 'info',
                        content:
                          'The Foundation Model is currently under preparation.'
                      })
                  : () => setSelected(option.key)
              }
            >
              {
                <option.icon
                  style={{
                    fontSize: '40px',
                    color: selected === option.key ? '#4096ff' : '#9d9d9d'
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
      </Grid>
    );

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
                  color: selected === option.key ? '#4096ff' : '#9d9d9d'
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
